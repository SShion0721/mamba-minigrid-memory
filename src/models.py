"""Actor-critic networks for MiniGrid PPO experiments."""

from __future__ import annotations

from typing import Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from src.cue import CUE_CLASS_COUNT, CUE_IGNORE_INDEX, extract_cue_targets_torch

try:
    from mamba_ssm import Mamba as MambaBlock
    try:
        from mamba_ssm import Mamba2 as Mamba2Block
    except Exception:
        Mamba2Block = None
    try:
        from mamba_ssm import Mamba3 as Mamba3Block
    except ImportError:
        try:
            from mamba_ssm.modules.mamba3 import Mamba3 as Mamba3Block
        except ImportError:
            Mamba3Block = None
    except Exception:
        Mamba3Block = None
except Exception as exc:  # pragma: no cover - exercised only without mamba_ssm
    MambaBlock = None
    Mamba2Block = None
    Mamba3Block = None
    MAMBA_IMPORT_ERROR = exc
else:
    MAMBA_IMPORT_ERROR = None


def layer_init(layer: nn.Linear, std: float = 1.0, bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _safe_categorical(logits: torch.Tensor) -> Categorical:
    """Create a Categorical distribution with NaN/Inf-safe logits.

    Mamba blocks can occasionally produce NaN due to numerical instability.
    This helper replaces NaN with a very large negative value (effectively
    zero probability) and clamps extreme values to prevent overflow.
    """
    safe = torch.nan_to_num(logits, nan=-1e8, posinf=1e8, neginf=-1e8)
    safe = safe.clamp(-1e9, 1e9)
    return Categorical(logits=safe)


class MiniGridSpatialEncoder(nn.Module):
    """Encode a 7x7 semantic MiniGrid view.

    The transformer path is kept for ablations. The default hybrid path is more
    task-shaped for MiniGrid Memory: local convolutions model walls/corridors,
    while a learned saliency pool extracts the visible object cue.
    """

    def __init__(
        self,
        d_model: int = 128,
        spatial_layers: int = 2,
        spatial_heads: int = 4,
        dropout: float = 0.0,
        encoder_type: str = "hybrid",
        slot_count: int = 0,
        slot_extractor: str = "query_pool",
        slot_iters: int = 3,
        slot_mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.spatial_layers = spatial_layers
        self.encoder_type = encoder_type
        self.slot_count = max(0, int(slot_count))
        if slot_extractor not in {"query_pool", "iterative"}:
            raise ValueError("slot_extractor must be one of: query_pool, iterative.")
        self.slot_extractor = slot_extractor
        self.slot_iters = max(1, int(slot_iters))

        self.obj_emb = nn.Embedding(32, 24)
        self.color_emb = nn.Embedding(16, 16)
        self.state_emb = nn.Embedding(8, 8)
        self.cell_proj = layer_init(nn.Linear(24 + 16 + 8, d_model))

        self.direction_emb = nn.Embedding(4, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, 49, d_model))
        if self.slot_count > 0:
            self.slot_queries = nn.Parameter(torch.zeros(1, self.slot_count, d_model))
            self.slot_source_norm = nn.LayerNorm(d_model)
            self.slot_norm = nn.LayerNorm(d_model)
            if self.slot_extractor == "iterative":
                slot_hidden = max(d_model, int(d_model * slot_mlp_ratio))
                self.slot_gru = nn.GRUCell(d_model, d_model)
                self.slot_update_norm = nn.LayerNorm(d_model)
                self.slot_mlp = nn.Sequential(
                    layer_init(nn.Linear(d_model, slot_hidden)),
                    nn.GELU(),
                    layer_init(nn.Linear(slot_hidden, d_model)),
                )
            else:
                self.slot_gru = None
                self.slot_update_norm = None
                self.slot_mlp = None
        else:
            self.register_parameter("slot_queries", None)
            self.slot_source_norm = None
            self.slot_norm = None
            self.slot_gru = None
            self.slot_update_norm = None
            self.slot_mlp = None

        if encoder_type == "transformer" and spatial_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=spatial_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.spatial = nn.TransformerEncoder(encoder_layer, num_layers=spatial_layers)
            self.out_norm = nn.LayerNorm(d_model)
            self.conv_blocks = None
            self.saliency = None
            self.summary = None
        elif encoder_type == "transformer":
            self.spatial = None
            self.flat = nn.Sequential(
                layer_init(nn.Linear(49 * d_model + d_model, 2 * d_model)),
                nn.GELU(),
                layer_init(nn.Linear(2 * d_model, d_model)),
                nn.GELU(),
            )
            self.conv_blocks = None
            self.saliency = None
            self.summary = None
        elif encoder_type == "hybrid":
            self.spatial = None
            self.flat = None
            self.conv_blocks = nn.ModuleList(
                [_SpatialConvBlock(d_model=d_model, dropout=dropout) for _ in range(max(1, spatial_layers))]
            )
            self.saliency = nn.Sequential(
                nn.LayerNorm(d_model),
                layer_init(nn.Linear(d_model, 1), std=0.01),
            )
            self.summary = nn.Sequential(
                nn.LayerNorm(3 * d_model),
                layer_init(nn.Linear(3 * d_model, 2 * d_model)),
                nn.GELU(),
                layer_init(nn.Linear(2 * d_model, d_model)),
                nn.GELU(),
            )
            self.out_norm = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"Unknown spatial encoder type: {encoder_type}")

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        if self.slot_queries is not None:
            nn.init.trunc_normal_(self.slot_queries, std=0.02)

    def forward(self, obs: torch.Tensor, direction: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.forward_tokens(obs, direction)
        return out

    def forward_tokens(
        self,
        obs: torch.Tensor,
        direction: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Accept obs [B, 7, 7, 3] or [B, T, 7, 7, 3]."""

        has_time = obs.ndim == 5
        if has_time:
            batch, seq_len = obs.shape[:2]
            obs = obs.reshape(batch * seq_len, *obs.shape[2:])
            if direction is not None:
                direction = direction.reshape(batch * seq_len, -1)

        num_items = obs.shape[0]
        direction_idx = _direction_index(direction, num_items, obs.device)

        obj = obs[..., 0].long().clamp_(0, 31)
        color = obs[..., 1].long().clamp_(0, 15)
        state = obs[..., 2].long().clamp_(0, 7)

        cells = torch.cat(
            [self.obj_emb(obj), self.color_emb(color), self.state_emb(state)],
            dim=-1,
        )
        cells = self.cell_proj(cells.reshape(num_items, 49, -1))
        cells = cells + self.pos_emb

        direction_token = self.direction_emb(direction_idx).unsqueeze(1)
        cls = self.cls_token.expand(num_items, -1, -1) + direction_token

        if self.spatial is None:
            if self.encoder_type == "transformer":
                out = self.flat(torch.cat([cells.reshape(num_items, -1), direction_token.squeeze(1)], dim=-1))
                slot_source = cells
            else:
                grid = cells.transpose(1, 2).reshape(num_items, self.d_model, 7, 7)
                for block in self.conv_blocks:
                    grid = block(grid)
                tokens = grid.flatten(2).transpose(1, 2)
                weights = torch.softmax(self.saliency(tokens).squeeze(-1), dim=-1).unsqueeze(-1)
                salient = (tokens * weights).sum(dim=1)
                mean = tokens.mean(dim=1)
                out = self.out_norm(self.summary(torch.cat([salient, mean, direction_token.squeeze(1)], dim=-1)))
                slot_source = tokens
        else:
            tokens = self.spatial(torch.cat([cls, cells], dim=1))
            out = self.out_norm(tokens[:, 0])
            slot_source = tokens[:, 1:]

        slots = self._extract_slots(slot_source, direction_token)

        if has_time:
            out = out.reshape(batch, seq_len, self.d_model)
            if slots is not None:
                slots = slots.reshape(batch, seq_len, self.slot_count, self.d_model)
        return out, slots

    def _extract_slots(self, tokens: torch.Tensor, direction_token: torch.Tensor) -> torch.Tensor | None:
        if self.slot_count <= 0:
            return None
        queries = self.slot_queries.expand(tokens.shape[0], -1, -1) + direction_token
        source = self.slot_source_norm(tokens)
        if self.slot_extractor == "iterative":
            slots = queries
            for _ in range(self.slot_iters):
                slots_prev = slots
                logits = torch.matmul(self.slot_norm(slots), source.transpose(-2, -1)) / math.sqrt(self.d_model)
                attn = torch.softmax(logits, dim=1)
                attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                updates = torch.matmul(attn, tokens)
                slots = self.slot_gru(
                    updates.reshape(-1, self.d_model),
                    slots_prev.reshape(-1, self.d_model),
                ).reshape_as(slots_prev)
                slots = slots + self.slot_mlp(self.slot_update_norm(slots))
            return self.slot_norm(slots)
        weights = torch.softmax(torch.matmul(queries, source.transpose(-2, -1)) / math.sqrt(self.d_model), dim=-1)
        return self.slot_norm(torch.matmul(weights, tokens))


class _SpatialConvBlock(nn.Module):
    """Small residual mixer for 7x7 symbolic grids."""

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.GroupNorm(1, d_model)
        self.depthwise = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.pointwise = nn.Conv2d(d_model, 2 * d_model, kernel_size=1)
        self.out = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(self.norm(x))
        y = F.gelu(y)
        value, gate = self.pointwise(y).chunk(2, dim=1)
        y = value * torch.sigmoid(gate)
        y = self.out(y)
        return x + self.dropout(y)


class EpisodicCueMemory(nn.Module):
    """Small episode-local cue memory built only from visible observations."""

    def __init__(
        self,
        d_model: int,
        memory_slots: int = 16,
        topk: int = 4,
        write_window: int = 12,
    ):
        super().__init__()
        self.d_model = d_model
        self.memory_slots = max(1, int(memory_slots))
        self.topk = max(1, int(topk))
        self.write_window = max(1, int(write_window))
        self.key_proj = layer_init(nn.Linear(d_model, d_model))
        self.value_emb = nn.Embedding(CUE_CLASS_COUNT, d_model)
        self.read_proj = layer_init(nn.Linear(d_model, d_model))
        self.gate = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            layer_init(nn.Linear(2 * d_model, d_model)),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.value_emb.weight, std=0.02)
        self.last_gate_mean = 0.0
        self.last_retrieval_entropy = 0.0
        self.last_write_rate = 0.0

    def init_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        return {
            "keys": torch.zeros(batch_size, self.memory_slots, self.d_model, device=device, dtype=dtype),
            "vals": torch.zeros(batch_size, self.memory_slots, self.d_model, device=device, dtype=dtype),
            "valid": torch.zeros(batch_size, self.memory_slots, device=device, dtype=torch.bool),
            "ptr": torch.zeros(batch_size, device=device, dtype=torch.long),
            "age": torch.zeros(batch_size, device=device, dtype=torch.long),
            "episode_cue": torch.full(
                (batch_size,),
                CUE_IGNORE_INDEX,
                device=device,
                dtype=torch.long,
            ),
        }

    @staticmethod
    def reset_state(state: dict[str, torch.Tensor] | None, done_mask: torch.Tensor) -> None:
        if state is None or done_mask.numel() == 0 or not done_mask.any():
            return
        for key in ("keys", "vals", "valid"):
            state[key][done_mask] = 0
        state["ptr"][done_mask] = 0
        state["age"][done_mask] = 0
        state["episode_cue"][done_mask] = CUE_IGNORE_INDEX

    @staticmethod
    def _reset_state_copy(
        state: dict[str, torch.Tensor],
        done_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if done_mask.numel() == 0 or not done_mask.any():
            return state
        next_state = dict(state)
        for key in ("keys", "vals", "valid"):
            value = state[key].clone()
            value[done_mask] = 0
            next_state[key] = value
        ptr = state["ptr"].clone()
        age = state["age"].clone()
        episode_cue = state["episode_cue"].clone()
        ptr[done_mask] = 0
        age[done_mask] = 0
        episode_cue[done_mask] = CUE_IGNORE_INDEX
        next_state["ptr"] = ptr
        next_state["age"] = age
        next_state["episode_cue"] = episode_cue
        return next_state

    def forward(
        self,
        step_tokens: torch.Tensor,
        obs_seq: torch.Tensor,
        episode_start_seq: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, seq_len, _ = step_tokens.shape
        state = self.init_state(
            batch,
            device=step_tokens.device,
            dtype=step_tokens.dtype,
        )
        cue_targets = extract_cue_targets_torch(obs_seq)
        outputs = []
        gate_stats = []
        entropy_stats = []
        write_stats = []
        for idx in range(seq_len):
            valid = None if valid_mask is None else valid_mask[:, idx].to(device=step_tokens.device).bool()
            out, state, gate_mean, entropy, write_rate = self.forward_step(
                step_tokens[:, idx],
                cue_targets[:, idx],
                episode_start_seq[:, idx],
                state,
                valid,
            )
            outputs.append(out)
            gate_stats.append(gate_mean)
            entropy_stats.append(entropy)
            write_stats.append(write_rate)
        if gate_stats and not (hasattr(torch, "compiler") and torch.compiler.is_compiling()):
            self.last_gate_mean = float(torch.stack(gate_stats).mean().detach().cpu())
            self.last_retrieval_entropy = float(torch.stack(entropy_stats).mean().detach().cpu())
            self.last_write_rate = float(torch.stack(write_stats).mean().detach().cpu())
        return torch.stack(outputs, dim=1)

    def forward_step(
        self,
        step_token: torch.Tensor,
        cue_target: torch.Tensor,
        episode_start: torch.Tensor,
        state: dict[str, torch.Tensor] | None,
        valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        if state is None:
            state = self.init_state(
                step_token.shape[0],
                device=step_token.device,
                dtype=step_token.dtype,
            )
        episode_start = episode_start.reshape(step_token.shape[0], -1)[:, 0].to(device=step_token.device) > 0.5
        state = self._reset_state_copy(state, episode_start)
        if valid is None:
            valid = torch.ones(step_token.shape[0], device=step_token.device, dtype=torch.bool)
        else:
            valid = valid.to(device=step_token.device).bool()

        retrieved, entropy = self._read(step_token, state)
        gate = self.gate(torch.cat([step_token, retrieved], dim=-1))
        enhanced = self.out_norm(step_token + gate * self.read_proj(retrieved))

        current_target = cue_target.to(device=step_token.device).long()
        has_new_cue = (state["episode_cue"] == CUE_IGNORE_INDEX) & (current_target != CUE_IGNORE_INDEX)
        in_write_window = state["age"] < self.write_window
        write_mask = valid & has_new_cue & in_write_window
        write_rate = write_mask.float().mean()
        if write_mask.any():
            state = dict(state)
            state["episode_cue"] = torch.where(write_mask, current_target, state["episode_cue"])
            state = self._write(state, step_token, current_target, write_mask)
        state["age"] = torch.where(valid, state["age"] + 1, state["age"])
        if not (hasattr(torch, "compiler") and torch.compiler.is_compiling()):
            self.last_gate_mean = float(gate.mean().detach().cpu())
            self.last_retrieval_entropy = float(entropy.detach().cpu())
            self.last_write_rate = float(write_rate.detach().cpu())
        return enhanced, state, gate.mean(), entropy, write_rate

    def _read(self, query: torch.Tensor, state: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        keys = state["keys"]
        vals = state["vals"]
        valid = state["valid"]
        query_key = F.normalize(self.key_proj(query), dim=-1)
        memory_key = F.normalize(keys, dim=-1)
        scores = torch.einsum("bd,bsd->bs", query_key, memory_key)
        scores = scores.masked_fill(~valid, -1.0e9)
        k = min(self.topk, self.memory_slots)
        top_scores, top_idx = torch.topk(scores, k=k, dim=-1)
        attn = torch.softmax(top_scores, dim=-1)
        gathered = vals.gather(-2, top_idx.unsqueeze(-1).expand(-1, -1, vals.shape[-1]))
        retrieved = (gathered * attn.unsqueeze(-1)).sum(dim=-2)
        any_valid = valid.any(dim=-1, keepdim=True)
        retrieved = torch.where(any_valid, retrieved, torch.zeros_like(retrieved))
        entropy = -(attn * attn.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = torch.where(any_valid.squeeze(-1), entropy, torch.zeros_like(entropy))
        return retrieved, entropy.mean()

    def _write(
        self,
        state: dict[str, torch.Tensor],
        query: torch.Tensor,
        cue_target: torch.Tensor,
        write_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        rows = torch.nonzero(write_mask, as_tuple=False).squeeze(-1)
        if rows.numel() == 0:
            return state
        ptr = state["ptr"][rows]
        key = F.normalize(self.key_proj(query.detach()), dim=-1)
        value = self.value_emb(cue_target.clamp(min=0, max=CUE_CLASS_COUNT - 1))
        next_state = dict(state)
        keys = state["keys"].clone()
        vals = state["vals"].clone()
        valid = state["valid"].clone()
        ptr_state = state["ptr"].clone()
        keys[rows, ptr] = key[rows].to(dtype=keys.dtype)
        vals[rows, ptr] = value[rows].to(dtype=vals.dtype)
        valid[rows, ptr] = True
        ptr_state[rows] = (ptr + 1) % self.memory_slots
        next_state["keys"] = keys
        next_state["vals"] = vals
        next_state["valid"] = valid
        next_state["ptr"] = ptr_state
        return next_state


class TokenEncoder(nn.Module):
    """Build trajectory tokens from MiniGrid frame, direction, and side inputs."""

    def __init__(
        self,
        action_dim: int,
        d_model: int,
        spatial_layers: int = 2,
        spatial_heads: int = 4,
        dropout: float = 0.0,
        spatial_encoder: str = "hybrid",
        slot_count: int = 0,
        slot_extractor: str = "query_pool",
        slot_iters: int = 3,
        slot_mlp_ratio: float = 2.0,
        temporal_token_mode: str = "flatten",
        memory_kind: str = "none",
        memory_slots: int = 16,
        memory_topk: int = 4,
        memory_write_window: int = 12,
    ):
        super().__init__()
        self.slot_count = max(0, int(slot_count))
        if temporal_token_mode not in {"flatten", "fuse"}:
            raise ValueError("temporal_token_mode must be one of: flatten, fuse.")
        if memory_kind not in {"none", "episodic_cue"}:
            raise ValueError("memory_kind must be one of: none, episodic_cue.")
        self.temporal_token_mode = temporal_token_mode
        self.tokens_per_step = self.slot_count + 1 if temporal_token_mode == "flatten" else 1
        self.obs_encoder = MiniGridSpatialEncoder(
            d_model=d_model,
            spatial_layers=spatial_layers,
            spatial_heads=spatial_heads,
            dropout=dropout,
            encoder_type=spatial_encoder,
            slot_count=self.slot_count,
            slot_extractor=slot_extractor,
            slot_iters=slot_iters,
            slot_mlp_ratio=slot_mlp_ratio,
        )
        self.action_proj = layer_init(nn.Linear(action_dim, d_model))
        self.reward_proj = layer_init(nn.Linear(1, d_model))
        self.start_proj = layer_init(nn.Linear(1, d_model))
        self.token_norm = nn.LayerNorm(d_model)
        if self.slot_count > 0 and temporal_token_mode == "fuse":
            fuse_heads = spatial_heads if d_model % max(1, spatial_heads) == 0 else 1
            self.slot_fuse = nn.MultiheadAttention(d_model, fuse_heads, dropout=dropout, batch_first=True)
            self.slot_fuse_norm = nn.LayerNorm(d_model)
        else:
            self.slot_fuse = None
            self.slot_fuse_norm = None
        self.memory = (
            EpisodicCueMemory(
                d_model=d_model,
                memory_slots=memory_slots,
                topk=memory_topk,
                write_window=memory_write_window,
            )
            if memory_kind == "episodic_cue"
            else None
        )

    def forward(
        self,
        obs_seq: torch.Tensor,
        direction_seq: torch.Tensor,
        prev_action_seq: torch.Tensor,
        prev_reward_seq: torch.Tensor,
        episode_start_seq: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        return_valid_mask: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        spatial, slots = self.obs_encoder.forward_tokens(obs_seq, direction_seq)
        side = (
            self.action_proj(prev_action_seq.float())
            + self.reward_proj(prev_reward_seq.float())
            + self.start_proj(episode_start_seq.float())
        )
        global_token = spatial + side
        if slots is None or self.temporal_token_mode == "fuse":
            x = global_token if slots is None else self._fuse_step_slots(global_token, slots + side.unsqueeze(-2))
            token_valid_mask = valid_mask
        else:
            slot_tokens = slots + side.unsqueeze(-2)
            x = torch.cat([slot_tokens, global_token.unsqueeze(-2)], dim=-2)
            x = self.token_norm(x)
            if self.memory is not None:
                enhanced_global = self.memory(x[:, :, self.slot_count], obs_seq, episode_start_seq, valid_mask)
                x = torch.cat([x[:, :, : self.slot_count], enhanced_global.unsqueeze(-2)], dim=-2)
            batch, seq_len, tokens_per_step, d_model = x.shape
            x = x.reshape(batch, seq_len * tokens_per_step, d_model)
            token_valid_mask = None
            if valid_mask is not None:
                token_valid_mask = (
                    valid_mask.to(device=x.device)
                    .unsqueeze(-1)
                    .expand(-1, -1, tokens_per_step)
                    .reshape(batch, seq_len * tokens_per_step)
                )
            x = _apply_valid_mask(x, token_valid_mask)
            if return_valid_mask:
                return x, token_valid_mask
            return x
        x = self.token_norm(x)
        if self.memory is not None:
            x = self.memory(x, obs_seq, episode_start_seq, valid_mask)
        x = _apply_valid_mask(x, token_valid_mask)
        if return_valid_mask:
            return x, token_valid_mask
        return x

    def _fuse_step_slots(self, global_token: torch.Tensor, slot_tokens: torch.Tensor) -> torch.Tensor:
        if self.slot_fuse is None or self.slot_fuse_norm is None:
            return global_token
        batch, seq_len, slot_count, d_model = slot_tokens.shape
        q = global_token.reshape(batch * seq_len, 1, d_model)
        kv = slot_tokens.reshape(batch * seq_len, slot_count, d_model)
        fused, _ = self.slot_fuse(q, kv, kv, need_weights=False)
        fused = self.slot_fuse_norm(q + fused)
        return fused.reshape(batch, seq_len, d_model)

    def init_memory_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor] | None:
        if self.memory is None:
            return None
        return self.memory.init_state(batch_size, device=device, dtype=dtype)

    def reset_memory_state(self, state: dict[str, torch.Tensor] | None, done_mask: torch.Tensor) -> None:
        if self.memory is not None:
            self.memory.reset_state(state, done_mask)

    def forward_step(
        self,
        obs: torch.Tensor,
        direction: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        episode_start: torch.Tensor,
        memory_state: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        spatial, slots = self.obs_encoder.forward_tokens(obs.unsqueeze(1), direction.unsqueeze(1))
        side = (
            self.action_proj(prev_action.unsqueeze(1).float())
            + self.reward_proj(prev_reward.unsqueeze(1).float())
            + self.start_proj(episode_start.unsqueeze(1).float())
        )
        global_token = spatial + side
        if slots is None or self.temporal_token_mode == "fuse":
            x = global_token if slots is None else self._fuse_step_slots(global_token, slots + side.unsqueeze(-2))
            x = self.token_norm(x).squeeze(1)
            if self.memory is not None:
                cue_target = extract_cue_targets_torch(obs)
                x, memory_state, _, _, _ = self.memory.forward_step(x, cue_target, episode_start, memory_state)
            return x.unsqueeze(1), memory_state

        slot_tokens = slots + side.unsqueeze(-2)
        x = torch.cat([slot_tokens, global_token.unsqueeze(-2)], dim=-2)
        x = self.token_norm(x).squeeze(1)
        if self.memory is not None:
            cue_target = extract_cue_targets_torch(obs)
            enhanced_global, memory_state, _, _, _ = self.memory.forward_step(
                x[:, self.slot_count],
                cue_target,
                episode_start,
                memory_state,
            )
            x = torch.cat([x[:, : self.slot_count], enhanced_global.unsqueeze(1)], dim=1)
        return x, memory_state

    def decision_tokens(self, x: torch.Tensor, time_steps: int) -> torch.Tensor:
        if self.tokens_per_step == 1:
            return x[:, :time_steps]
        return x[:, self.slot_count :: self.tokens_per_step][:, :time_steps]

    def expand_lengths(self, lengths: torch.Tensor | None) -> torch.Tensor | None:
        if lengths is None:
            return None
        return lengths * self.tokens_per_step


class GRUGate(nn.Module):
    """用于强化学习 Transformer 的门控机制 (Stabilizing RL Transformers)"""
    def __init__(self, d_model: int):
        super().__init__()
        self.linear_w_r = nn.Linear(d_model, d_model, bias=False)
        self.linear_u_r = nn.Linear(d_model, d_model, bias=False)
        self.linear_w_z = nn.Linear(d_model, d_model, bias=False)
        self.linear_u_z = nn.Linear(d_model, d_model, bias=False)
        self.linear_w_g = nn.Linear(d_model, d_model, bias=False)
        self.linear_u_g = nn.Linear(d_model, d_model, bias=False)
        self.bias_r = nn.Parameter(torch.zeros(d_model))
        self.bias_z = nn.Parameter(torch.zeros(d_model))
        self.bias_g = nn.Parameter(torch.zeros(d_model))
        
        # 将 z (update gate) 的偏置初始化为较大的负数，
        # 这确保了网络初始化时几乎是 Identity Mapping (x_new ≈ x_old)，这对 PPO 极度重要！
        nn.init.constant_(self.bias_z, -2.0)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: prev_state (残差连接), y: new_info (如 attention 输出)
        r = torch.sigmoid(self.linear_w_r(y) + self.linear_u_r(x) + self.bias_r)
        z = torch.sigmoid(self.linear_w_z(y) + self.linear_u_z(x) + self.bias_z)
        g = torch.tanh(self.linear_w_g(y) + self.linear_u_g(r * x) + self.bias_g)
        return (1.0 - z) * x + z * g


class GatedAttentionBlock(nn.Module):
    """结合 FlashAttention 和 GRU 门控的 Block"""
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}.")
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.gate1 = GRUGate(d_model)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            layer_init(nn.Linear(d_model, 4 * d_model)),
            nn.GELU(),
            layer_init(nn.Linear(4 * d_model, d_model))
        )
        self.gate2 = GRUGate(d_model)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

    def forward(
        self,
        x: torch.Tensor,
        *,
        is_causal: bool = True,
        alibi_slopes: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1. 快速注意力计算
        B, T, C = x.shape
        qkv = self.qkv(self.norm1(x))
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 调用 PyTorch 原生的高效 SDPA (底层自动使用 FlashAttention)
        attn_mask = None
        if alibi_slopes is not None:
            attn_mask = _alibi_attention_mask(T, alibi_slopes, q.device, q.dtype, causal=is_causal)
            is_causal = False
        if valid_mask is not None:
            pad_bias = _padding_attention_bias(valid_mask, q.device, q.dtype)
            if attn_mask is None:
                attn_mask = _causal_attention_bias(T, q.device, q.dtype) if is_causal else 0.0
            attn_mask = attn_mask + pad_bias
            is_causal = False
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        attn_out = self.out_proj(attn_out)
        
        # 使用 GRU 门控替代原来的简单残差 x = x + attn_out
        x = self.gate1(x, attn_out)
        
        # 2. MLP 层
        mlp_out = self.mlp(self.norm2(x))
        x = self.gate2(x, mlp_out)
        
        return _apply_valid_mask(torch.nan_to_num(x), valid_mask)

    def forward_step(
        self,
        x: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        alibi_slopes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape
        if T != 1:
            raise ValueError("GatedAttentionBlock.forward_step expects a single token.")

        cache_k, cache_v, lengths = cache
        qkv = self.qkv(self.norm1(x))
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2).squeeze(2)
        v = v.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2).squeeze(2)

        max_len = cache_k.shape[2]
        full = lengths >= max_len
        if bool(full.any()):
            cache_k[full, :, :-1] = cache_k[full, :, 1:].clone()
            cache_v[full, :, :-1] = cache_v[full, :, 1:].clone()

        rows = torch.arange(B, device=x.device)
        write_pos = torch.minimum(lengths, torch.full_like(lengths, max_len - 1))
        cache_k[rows, :, write_pos] = k
        cache_v[rows, :, write_pos] = v
        lengths = torch.clamp(lengths + 1, max=max_len)

        scores = torch.matmul(q, cache_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        positions = torch.arange(max_len, device=x.device)
        valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
        scores = scores.masked_fill(~valid[:, None, None, :], torch.finfo(scores.dtype).min)
        if alibi_slopes is not None:
            age = (lengths[:, None] - 1 - positions[None, :]).clamp_min(0)
            scores = scores - alibi_slopes.to(dtype=scores.dtype)[None, :, None, None] * age[:, None, None, :]

        attn = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        attn_out = torch.matmul(attn, cache_v).transpose(1, 2).contiguous().view(B, 1, C)
        attn_out = self.out_proj(attn_out)

        x = self.gate1(x, attn_out)
        mlp_out = self.mlp(self.norm2(x))
        x = self.gate2(x, mlp_out)
        return x, (cache_k, cache_v, lengths)
    

def _init_cue_head(module: nn.Module, d_model: int, aux_recall: bool) -> None:
    module.cue_head = layer_init(nn.Linear(d_model, CUE_CLASS_COUNT), std=0.01) if aux_recall else None


def _actor_critic_output(
    module: nn.Module,
    features: torch.Tensor,
    *,
    return_aux: bool,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    logits = _mask_logits(module, module.actor(features))
    values = module.critic(features).squeeze(-1)
    if return_aux:
        cue_head = getattr(module, "cue_head", None)
        aux_logits = cue_head(features) if cue_head is not None else None
        return logits, values, aux_logits
    return logits, values


class MLPActorCritic(nn.Module):
    """Feedforward PPO baseline with spatial attention but no temporal memory."""

    def __init__(
        self,
        action_dim: int = 7,
        d_model: int = 128,
        spatial_layers: int = 2,
        spatial_heads: int = 4,
        dropout: float = 0.0,
        valid_actions: list[int] | None = None,
        spatial_encoder: str = "hybrid",
        slot_count: int = 0,
        aux_recall: bool = False,
    ):
        super().__init__()
        self.action_dim = action_dim
        _register_action_mask(self, action_dim, valid_actions)
        self.encoder = MiniGridSpatialEncoder(
            d_model=d_model,
            spatial_layers=spatial_layers,
            spatial_heads=spatial_heads,
            dropout=dropout,
            encoder_type=spatial_encoder,
        )
        self.action_proj = layer_init(nn.Linear(action_dim, d_model))
        self.reward_proj = layer_init(nn.Linear(1, d_model))
        self.start_proj = layer_init(nn.Linear(1, d_model))
        self.shared = nn.Sequential(
            nn.LayerNorm(d_model),
            layer_init(nn.Linear(d_model, d_model)),
            nn.GELU(),
            layer_init(nn.Linear(d_model, d_model)),
            nn.GELU(),
        )
        self.actor = layer_init(nn.Linear(d_model, action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(d_model, 1), std=1.0)
        _init_cue_head(self, d_model, aux_recall)

    def forward(
        self,
        obs: torch.Tensor,
        direction: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        episode_start: torch.Tensor,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        x = (
            self.encoder(obs, direction)
            + self.action_proj(prev_action.float())
            + self.reward_proj(prev_reward.float())
            + self.start_proj(episode_start.float())
        )
        x = self.shared(x)
        return _actor_critic_output(self, x, return_aux=return_aux)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        direction: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        episode_start: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs, direction, prev_action, prev_reward, episode_start)
        dist = _safe_categorical(logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


class LSTMActorCritic(nn.Module):
    """LSTM recurrent PPO baseline with the same spatial encoder."""

    def __init__(
        self,
        action_dim: int = 7,
        d_model: int = 128,
        n_layers: int = 1,
        spatial_layers: int = 2,
        spatial_heads: int = 4,
        dropout: float = 0.0,
        valid_actions: list[int] | None = None,
        spatial_encoder: str = "hybrid",
        slot_count: int = 0,
        slot_extractor: str = "query_pool",
        slot_iters: int = 3,
        slot_mlp_ratio: float = 2.0,
        temporal_token_mode: str = "flatten",
        memory_kind: str = "none",
        memory_slots: int = 16,
        memory_topk: int = 4,
        memory_write_window: int = 12,
        aux_recall: bool = False,
    ):
        super().__init__()
        self.action_dim = action_dim
        _register_action_mask(self, action_dim, valid_actions)
        self.token_encoder = TokenEncoder(
            action_dim=action_dim,
            d_model=d_model,
            spatial_layers=spatial_layers,
            spatial_heads=spatial_heads,
            dropout=dropout,
            spatial_encoder=spatial_encoder,
            slot_count=slot_count,
            slot_extractor=slot_extractor,
            slot_iters=slot_iters,
            slot_mlp_ratio=slot_mlp_ratio,
            temporal_token_mode=temporal_token_mode,
            memory_kind=memory_kind,
            memory_slots=memory_slots,
            memory_topk=memory_topk,
            memory_write_window=memory_write_window,
        )
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.actor = layer_init(nn.Linear(d_model, action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(d_model, 1), std=1.0)
        _init_cue_head(self, d_model, aux_recall)

    def forward(
        self,
        obs_seq: torch.Tensor,
        direction_seq: torch.Tensor,
        prev_action_seq: torch.Tensor,
        prev_reward_seq: torch.Tensor,
        episode_start_seq: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        time_steps = obs_seq.shape[1]
        x, token_valid_mask = self.token_encoder(
            obs_seq,
            direction_seq,
            prev_action_seq,
            prev_reward_seq,
            episode_start_seq,
            valid_mask,
            return_valid_mask=True,
        )
        token_lengths = self.token_encoder.expand_lengths(_resolve_lengths(valid_mask, lengths, time_steps))
        token_seq_len = x.shape[1]
        if token_lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x,
                token_lengths.detach().to("cpu"),
                batch_first=True,
                enforce_sorted=False,
            )
            x, _ = self.lstm(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=token_seq_len)
        else:
            x, _ = self.lstm(x)
        x = self.norm(x)
        x = _apply_valid_mask(x, token_valid_mask)
        x = self.token_encoder.decision_tokens(x, time_steps)
        return _actor_critic_output(self, x, return_aux=return_aux)

    def get_action_and_value(
        self,
        obs_seq: torch.Tensor,
        direction_seq: torch.Tensor,
        prev_action_seq: torch.Tensor,
        prev_reward_seq: torch.Tensor,
        episode_start_seq: torch.Tensor,
        action: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(
            obs_seq,
            direction_seq,
            prev_action_seq,
            prev_reward_seq,
            episode_start_seq,
            valid_mask=valid_mask,
            lengths=lengths,
        )
        last_logits = _gather_last_valid(logits, valid_mask)
        last_values = _gather_last_valid(values, valid_mask)
        dist = _safe_categorical(last_logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), last_values


class GRUActorCritic(nn.Module):
    """GRU recurrent PPO baseline with the shared spatial/token encoder."""

    def __init__(
        self,
        action_dim: int = 7,
        d_model: int = 128,
        n_layers: int = 1,
        spatial_layers: int = 2,
        spatial_heads: int = 4,
        dropout: float = 0.0,
        valid_actions: list[int] | None = None,
        spatial_encoder: str = "hybrid",
        slot_count: int = 0,
        slot_extractor: str = "query_pool",
        slot_iters: int = 3,
        slot_mlp_ratio: float = 2.0,
        temporal_token_mode: str = "flatten",
        memory_kind: str = "none",
        memory_slots: int = 16,
        memory_topk: int = 4,
        memory_write_window: int = 12,
        aux_recall: bool = False,
    ):
        super().__init__()
        self.action_dim = action_dim
        _register_action_mask(self, action_dim, valid_actions)
        self.token_encoder = TokenEncoder(
            action_dim=action_dim,
            d_model=d_model,
            spatial_layers=spatial_layers,
            spatial_heads=spatial_heads,
            dropout=dropout,
            spatial_encoder=spatial_encoder,
            slot_count=slot_count,
            slot_extractor=slot_extractor,
            slot_iters=slot_iters,
            slot_mlp_ratio=slot_mlp_ratio,
            temporal_token_mode=temporal_token_mode,
            memory_kind=memory_kind,
            memory_slots=memory_slots,
            memory_topk=memory_topk,
            memory_write_window=memory_write_window,
        )
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.actor = layer_init(nn.Linear(d_model, action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(d_model, 1), std=1.0)
        _init_cue_head(self, d_model, aux_recall)

    def forward(
        self,
        obs_seq: torch.Tensor,
        direction_seq: torch.Tensor,
        prev_action_seq: torch.Tensor,
        prev_reward_seq: torch.Tensor,
        episode_start_seq: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        time_steps = obs_seq.shape[1]
        x, token_valid_mask = self.token_encoder(
            obs_seq,
            direction_seq,
            prev_action_seq,
            prev_reward_seq,
            episode_start_seq,
            valid_mask,
            return_valid_mask=True,
        )
        token_lengths = self.token_encoder.expand_lengths(_resolve_lengths(valid_mask, lengths, time_steps))
        token_seq_len = x.shape[1]
        if token_lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x,
                token_lengths.detach().to("cpu"),
                batch_first=True,
                enforce_sorted=False,
            )
            x, _ = self.gru(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=token_seq_len)
        else:
            x, _ = self.gru(x)
        x = self.norm(x)
        x = _apply_valid_mask(x, token_valid_mask)
        x = self.token_encoder.decision_tokens(x, time_steps)
        return _actor_critic_output(self, x, return_aux=return_aux)

    def get_action_and_value(
        self,
        obs_seq: torch.Tensor,
        direction_seq: torch.Tensor,
        prev_action_seq: torch.Tensor,
        prev_reward_seq: torch.Tensor,
        episode_start_seq: torch.Tensor,
        action: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(
            obs_seq,
            direction_seq,
            prev_action_seq,
            prev_reward_seq,
            episode_start_seq,
            valid_mask=valid_mask,
            lengths=lengths,
        )
        last_logits = _gather_last_valid(logits, valid_mask)
        last_values = _gather_last_valid(values, valid_mask)
        dist = _safe_categorical(last_logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), last_values


class MambaActorCritic(nn.Module):
    """Spatial-attention + temporal-Mamba actor critic."""

    def __init__(
        self,
        action_dim: int = 7,
        d_model: int = 128,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        variant: str = "mamba",
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 64,
        rope_fraction: float = 0.5,
        spatial_layers: int = 2,
        spatial_heads: int = 4,
        dropout: float = 0.0,
        valid_actions: list[int] | None = None,
        spatial_encoder: str = "hybrid",
        slot_count: int = 0,
        slot_extractor: str = "query_pool",
        slot_iters: int = 3,
        slot_mlp_ratio: float = 2.0,
        temporal_token_mode: str = "flatten",
        memory_kind: str = "none",
        memory_slots: int = 16,
        memory_topk: int = 4,
        memory_write_window: int = 12,
        aux_recall: bool = False,
    ):
        super().__init__()
        block_cls = _resolve_mamba_block(variant)
        if block_cls is None:
            raise ImportError(
                f"mamba_ssm with variant={variant!r} is required for MambaActorCritic. "
                "Install it after CUDA PyTorch "
                "with: pip install mamba-ssm[causal-conv1d] --no-build-isolation"
            ) from MAMBA_IMPORT_ERROR

        self.action_dim = action_dim
        self.variant = variant
        self.residual_scale = 1.0 / max(n_layers, 1) ** 0.5
        _register_action_mask(self, action_dim, valid_actions)
        self.token_encoder = TokenEncoder(
            action_dim=action_dim,
            d_model=d_model,
            spatial_layers=spatial_layers,
            spatial_heads=spatial_heads,
            dropout=dropout,
            spatial_encoder=spatial_encoder,
            slot_count=slot_count,
            slot_extractor=slot_extractor,
            slot_iters=slot_iters,
            slot_mlp_ratio=slot_mlp_ratio,
            temporal_token_mode=temporal_token_mode,
            memory_kind=memory_kind,
            memory_slots=memory_slots,
            memory_topk=memory_topk,
            memory_write_window=memory_write_window,
        )
        self.blocks = nn.ModuleList(
            [
                _build_mamba_block(
                    variant=variant,
                    block_cls=block_cls,
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    headdim=headdim,
                    ngroups=ngroups,
                    chunk_size=chunk_size,
                    rope_fraction=rope_fraction,
                    dropout=dropout,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(n_layers)
            ]
        )
        self.block_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.actor = layer_init(nn.Linear(d_model, action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(d_model, 1), std=1.0)
        _init_cue_head(self, d_model, aux_recall)

        #增加mamba梯度稳定性：在每个block输出后添加残差连接和LayerNorm，并在输出前对block输出进行clamp，防止数值过大导致NaN。

    def forward(
        self,
        obs_seq: torch.Tensor,
        direction_seq: torch.Tensor,
        prev_action_seq: torch.Tensor,
        prev_reward_seq: torch.Tensor,
        episode_start_seq: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        time_steps = obs_seq.shape[1]
        x, token_valid_mask = self.token_encoder(
            obs_seq,
            direction_seq,
            prev_action_seq,
            prev_reward_seq,
            episode_start_seq,
            valid_mask,
            return_valid_mask=True,
        )
        for norm, block in zip(self.block_norms, self.blocks):
            out = block(norm(x))
            _raise_if_nonfinite(out, "Mamba block output")
            x = x + self.residual_scale * out
        x = self.norm(x)
        x = _apply_valid_mask(x, token_valid_mask)
        x = self.token_encoder.decision_tokens(x, time_steps)
        return _actor_critic_output(self, x, return_aux=return_aux)

    def init_inference_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> list[tuple[torch.Tensor, ...]] | dict[str, object]:
        """Allocate Mamba recurrent caches for one-token rollout inference."""

        states: list[tuple[torch.Tensor, ...]] = []
        for block in self.blocks:
            if not hasattr(block, "allocate_inference_cache"):
                raise RuntimeError(f"{type(block).__name__} does not support inference caching.")
            cache = block.allocate_inference_cache(batch_size, 1, dtype=dtype)
            cache = tuple(t.to(device=device) if device is not None else t for t in cache)
            states.append(cache)
        memory_state = self.token_encoder.init_memory_state(
            batch_size,
            device=device or next(self.parameters()).device,
            dtype=dtype or next(self.parameters()).dtype,
        )
        if memory_state is not None:
            return {"blocks": states, "memory": memory_state}
        return states

    def reset_inference_state(
        self,
        inference_state: list[tuple[torch.Tensor, ...]] | None,
        done_mask: torch.Tensor,
    ) -> None:
        if inference_state is None or done_mask.numel() == 0 or not done_mask.any():
            return
        memory_state = None
        if isinstance(inference_state, dict):
            memory_state = inference_state.get("memory")
            inference_state = inference_state["blocks"]
        for cache in inference_state:
            for tensor in cache:
                tensor[done_mask] = 0
        self.token_encoder.reset_memory_state(memory_state, done_mask)

    def get_action_and_value_step(
        self,
        obs: torch.Tensor,
        direction: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        episode_start: torch.Tensor,
        inference_state: list[tuple[torch.Tensor, ...]] | dict[str, object],
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, ...]] | dict[str, object]]:
        memory_state = None
        current_state = inference_state
        if isinstance(inference_state, dict):
            memory_state = inference_state.get("memory")
            current_state = inference_state["blocks"]
        x_tokens, memory_state = self.token_encoder.forward_step(
            obs,
            direction,
            prev_action,
            prev_reward,
            episode_start,
            memory_state,
        )
        x = x_tokens[:, -1:]
        for token_idx in range(x_tokens.shape[1]):
            x = x_tokens[:, token_idx : token_idx + 1]
            new_state: list[tuple[torch.Tensor, ...]] = []
            for cache, norm, block in zip(current_state, self.block_norms, self.blocks):
                out = block.step(norm(x), *cache)
                block_out = out[0]
                block_cache = tuple(out[1:])
                _raise_if_nonfinite(block_out, "Mamba step output")
                x = x + self.residual_scale * block_out
                new_state.append(block_cache)
            current_state = new_state

        x = self.norm(x)
        logits = _mask_logits(self, self.actor(x[:, -1]))
        value = self.critic(x[:, -1]).squeeze(-1)
        dist = _safe_categorical(logits)
        if action is None:
            action = dist.sample()
        packed_state = {"blocks": current_state, "memory": memory_state} if memory_state is not None else current_state
        return action, dist.log_prob(action), dist.entropy(), value, packed_state

    def get_action_and_value(
        self,
        obs_seq: torch.Tensor,
        direction_seq: torch.Tensor,
        prev_action_seq: torch.Tensor,
        prev_reward_seq: torch.Tensor,
        episode_start_seq: torch.Tensor,
        action: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(
            obs_seq,
            direction_seq,
            prev_action_seq,
            prev_reward_seq,
            episode_start_seq,
            valid_mask=valid_mask,
            lengths=lengths,
        )
        last_logits = _gather_last_valid(logits, valid_mask)
        last_values = _gather_last_valid(values, valid_mask)
        dist = _safe_categorical(last_logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), last_values


class FastGatedAttentionActorCritic(nn.Module):
    """Gated Causal Attention Actor Critic for RL"""
    def __init__(
        self,
        action_dim: int = 7,
        d_model: int = 128,
        n_layers: int = 2,     
        n_heads: int = 4,
        context_len: int = 128,
        spatial_layers: int = 2,
        spatial_heads: int = 4,
        dropout: float = 0.0,
        valid_actions: list[int] | None = None,
        spatial_encoder: str = "hybrid",
        position_mode: str = "learned",
        slot_count: int = 0,
        slot_extractor: str = "query_pool",
        slot_iters: int = 3,
        slot_mlp_ratio: float = 2.0,
        temporal_token_mode: str = "flatten",
        memory_kind: str = "none",
        memory_slots: int = 16,
        memory_topk: int = 4,
        memory_write_window: int = 12,
        aux_recall: bool = False,
    ):
        super().__init__()
        if position_mode not in {"learned", "none", "alibi"}:
            raise ValueError("position_mode must be one of: learned, none, alibi.")
        self.action_dim = action_dim
        _register_action_mask(self, action_dim, valid_actions)
        self.position_mode = position_mode
        self.n_heads = n_heads
        self.d_model = d_model
        
        self.token_encoder = TokenEncoder(
            action_dim=action_dim, d_model=d_model, spatial_layers=spatial_layers,
            spatial_heads=spatial_heads, dropout=dropout, spatial_encoder=spatial_encoder,
            slot_count=slot_count,
            slot_extractor=slot_extractor,
            slot_iters=slot_iters,
            slot_mlp_ratio=slot_mlp_ratio,
            temporal_token_mode=temporal_token_mode,
            memory_kind=memory_kind,
            memory_slots=memory_slots,
            memory_topk=memory_topk,
            memory_write_window=memory_write_window,
        )
        self.time_context_len = context_len
        self.context_len = context_len * self.token_encoder.tokens_per_step
        
        if self.position_mode == "learned":
            self.temporal_pos = nn.Parameter(torch.zeros(1, self.context_len, d_model))
            nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        else:
            self.register_parameter("temporal_pos", None)
        if self.position_mode == "alibi":
            self.register_buffer("alibi_slopes", _build_alibi_slopes(n_heads), persistent=False)
        else:
            self.alibi_slopes = None
        
        # Gated Attention Blocks
        self.blocks = nn.ModuleList([
            GatedAttentionBlock(d_model=d_model, n_heads=n_heads)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.actor = layer_init(nn.Linear(d_model, action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(d_model, 1), std=1.0)
        _init_cue_head(self, d_model, aux_recall)

    def forward(
        self,
        obs_seq,
        direction_seq,
        prev_action_seq,
        prev_reward_seq,
        episode_start_seq,
        valid_mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        
        time_steps = obs_seq.shape[1]
        x, token_valid_mask = self.token_encoder(
            obs_seq,
            direction_seq,
            prev_action_seq,
            prev_reward_seq,
            episode_start_seq,
            valid_mask,
            return_valid_mask=True,
        )
        seq_len = x.shape[1]
        
        if seq_len > self.context_len:
            raise ValueError(f"Sequence length {seq_len} exceeds context_len {self.context_len}.")
            
        if self.temporal_pos is not None:
            x = x + self.temporal_pos[:, -seq_len:]
            x = _apply_valid_mask(x, token_valid_mask)
        
        # 前向传播过 Gated Blocks
        for block in self.blocks:
            x = block(x, is_causal=True, alibi_slopes=self.alibi_slopes, valid_mask=token_valid_mask)
            
        x = self.norm(x)
        x = _apply_valid_mask(x, token_valid_mask)
        x = self.token_encoder.decision_tokens(x, time_steps)
        return _actor_critic_output(self, x, return_aux=return_aux)

    def get_action_and_value(
        self,
        obs_seq,
        direction_seq,
        prev_action_seq,
        prev_reward_seq,
        episode_start_seq,
        action=None,
        valid_mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ):
        logits, values = self.forward(
            obs_seq,
            direction_seq,
            prev_action_seq,
            prev_reward_seq,
            episode_start_seq,
            valid_mask=valid_mask,
            lengths=lengths,
        )
        last_logits = _gather_last_valid(logits, valid_mask)
        last_values = _gather_last_valid(values, valid_mask)
        dist = _safe_categorical(last_logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), last_values

    def init_inference_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | dict[str, object]:
        if self.position_mode == "learned":
            raise RuntimeError("Stateful Gated Attention rollout requires --gated-attention-pos none or alibi.")
        param = next(self.parameters())
        device = device or param.device
        dtype = dtype or param.dtype
        states = []
        for block in self.blocks:
            cache_k = torch.zeros(
                batch_size,
                block.n_heads,
                self.context_len,
                block.head_dim,
                device=device,
                dtype=dtype,
            )
            cache_v = torch.zeros_like(cache_k)
            lengths = torch.zeros(batch_size, device=device, dtype=torch.long)
            states.append((cache_k, cache_v, lengths))
        memory_state = self.token_encoder.init_memory_state(batch_size, device=device, dtype=dtype)
        if memory_state is not None:
            return {"blocks": states, "memory": memory_state}
        return states

    def reset_inference_state(
        self,
        inference_state: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None,
        done_mask: torch.Tensor,
    ) -> None:
        if inference_state is None or done_mask.numel() == 0 or not done_mask.any():
            return
        memory_state = None
        if isinstance(inference_state, dict):
            memory_state = inference_state.get("memory")
            inference_state = inference_state["blocks"]
        for cache_k, cache_v, lengths in inference_state:
            cache_k[done_mask] = 0
            cache_v[done_mask] = 0
            lengths[done_mask] = 0
        self.token_encoder.reset_memory_state(memory_state, done_mask)

    def get_action_and_value_step(
        self,
        obs: torch.Tensor,
        direction: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        episode_start: torch.Tensor,
        inference_state: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | dict[str, object],
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | dict[str, object]]:
        memory_state = None
        current_state = inference_state
        if isinstance(inference_state, dict):
            memory_state = inference_state.get("memory")
            current_state = inference_state["blocks"]
        x_tokens, memory_state = self.token_encoder.forward_step(
            obs,
            direction,
            prev_action,
            prev_reward,
            episode_start,
            memory_state,
        )
        x = x_tokens[:, -1:]
        for token_idx in range(x_tokens.shape[1]):
            x = x_tokens[:, token_idx : token_idx + 1]
            new_state = []
            for block, cache in zip(self.blocks, current_state):
                x, cache = block.forward_step(x, cache, alibi_slopes=self.alibi_slopes)
                new_state.append(cache)
            current_state = new_state

        x = self.norm(x)
        logits = _mask_logits(self, self.actor(x[:, -1]))
        value = self.critic(x[:, -1]).squeeze(-1)
        dist = _safe_categorical(logits)
        if action is None:
            action = dist.sample()
        packed_state = {"blocks": current_state, "memory": memory_state} if memory_state is not None else current_state
        return action, dist.log_prob(action), dist.entropy(), value, packed_state
    
class AttentionActorCritic(nn.Module):
    """Spatial-attention + causal temporal-attention actor critic."""

    def __init__(
        self,
        action_dim: int = 7,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        context_len: int = 64,
        spatial_layers: int = 2,
        spatial_heads: int = 4,
        dropout: float = 0.0,
        valid_actions: list[int] | None = None,
        spatial_encoder: str = "hybrid",
        slot_count: int = 0,
        slot_extractor: str = "query_pool",
        slot_iters: int = 3,
        slot_mlp_ratio: float = 2.0,
        temporal_token_mode: str = "flatten",
        memory_kind: str = "none",
        memory_slots: int = 16,
        memory_topk: int = 4,
        memory_write_window: int = 12,
        aux_recall: bool = False,
    ):
        super().__init__()
        self.action_dim = action_dim
        _register_action_mask(self, action_dim, valid_actions)
        self.n_heads = n_heads
        self.token_encoder = TokenEncoder(
            action_dim=action_dim,
            d_model=d_model,
            spatial_layers=spatial_layers,
            spatial_heads=spatial_heads,
            dropout=dropout,
            spatial_encoder=spatial_encoder,
            slot_count=slot_count,
            slot_extractor=slot_extractor,
            slot_iters=slot_iters,
            slot_mlp_ratio=slot_mlp_ratio,
            temporal_token_mode=temporal_token_mode,
            memory_kind=memory_kind,
            memory_slots=memory_slots,
            memory_topk=memory_topk,
            memory_write_window=memory_write_window,
        )
        self.time_context_len = context_len
        self.context_len = context_len * self.token_encoder.tokens_per_step
        self.temporal_pos = nn.Parameter(torch.zeros(1, self.context_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.actor = layer_init(nn.Linear(d_model, action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(d_model, 1), std=1.0)
        _init_cue_head(self, d_model, aux_recall)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

    def forward(
        self,
        obs_seq: torch.Tensor,
        direction_seq: torch.Tensor,
        prev_action_seq: torch.Tensor,
        prev_reward_seq: torch.Tensor,
        episode_start_seq: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        time_steps = obs_seq.shape[1]
        x, token_valid_mask = self.token_encoder(
            obs_seq,
            direction_seq,
            prev_action_seq,
            prev_reward_seq,
            episode_start_seq,
            valid_mask,
            return_valid_mask=True,
        )
        seq_len = x.shape[1]
        if seq_len > self.context_len:
            raise ValueError(f"Sequence length {seq_len} exceeds context_len {self.context_len}.")
        x = x + self.temporal_pos[:, -seq_len:]
        x = _apply_valid_mask(x, token_valid_mask)
        mask = _temporal_attention_mask(token_valid_mask, seq_len, self.n_heads, x.device, x.dtype)
        if mask is None:
            mask = _causal_mask(seq_len, x.device)
        x = self.temporal(x, mask=mask)
        x = _apply_valid_mask(torch.nan_to_num(x), token_valid_mask)
        x = self.norm(x)
        x = _apply_valid_mask(x, token_valid_mask)
        x = self.token_encoder.decision_tokens(x, time_steps)
        return _actor_critic_output(self, x, return_aux=return_aux)

    def get_action_and_value(
        self,
        obs_seq: torch.Tensor,
        direction_seq: torch.Tensor,
        prev_action_seq: torch.Tensor,
        prev_reward_seq: torch.Tensor,
        episode_start_seq: torch.Tensor,
        action: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(
            obs_seq,
            direction_seq,
            prev_action_seq,
            prev_reward_seq,
            episode_start_seq,
            valid_mask=valid_mask,
            lengths=lengths,
        )
        last_logits = _gather_last_valid(logits, valid_mask)
        last_values = _gather_last_valid(values, valid_mask)
        dist = _safe_categorical(last_logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), last_values


def build_actor_critic(config: Any, action_dim: int) -> nn.Module:
    """Instantiate a policy from a config object or config dict."""

    cfg = config if not isinstance(config, dict) else _DictConfig(config)
    model_name = getattr(cfg, "model")
    d_model = getattr(cfg, "d_model", 128)
    spatial_layers = getattr(cfg, "spatial_layers", 2)
    spatial_heads = getattr(cfg, "spatial_heads", 4)
    spatial_encoder = getattr(cfg, "spatial_encoder", "transformer")
    dropout = getattr(cfg, "dropout", 0.0)
    valid_actions = _parse_valid_actions(getattr(cfg, "valid_actions", None))
    slot_count = getattr(cfg, "slot_count", 0)
    memory_kwargs = {
        "slot_extractor": getattr(cfg, "slot_extractor", "query_pool"),
        "slot_iters": getattr(cfg, "slot_iters", 3),
        "slot_mlp_ratio": getattr(cfg, "slot_mlp_ratio", 2.0),
        "temporal_token_mode": getattr(cfg, "temporal_token_mode", "flatten"),
        "memory_kind": getattr(cfg, "memory_kind", "none"),
        "memory_slots": getattr(cfg, "memory_slots", 16),
        "memory_topk": getattr(cfg, "memory_topk", 4),
        "memory_write_window": getattr(cfg, "memory_write_window", 12),
        "aux_recall": getattr(cfg, "aux_recall_coef", 0.0) > 0,
    }

    if model_name == "mlp":
        return MLPActorCritic(
            action_dim=action_dim,
            d_model=d_model,
            spatial_layers=spatial_layers,
            spatial_heads=spatial_heads,
            dropout=dropout,
            valid_actions=valid_actions,
            spatial_encoder=spatial_encoder,
            aux_recall=memory_kwargs["aux_recall"],
        )
    if model_name == "lstm":
        return LSTMActorCritic(
            action_dim=action_dim,
            d_model=d_model,
            n_layers=getattr(cfg, "lstm_layers", 1),
            spatial_layers=spatial_layers,
            spatial_heads=spatial_heads,
            dropout=dropout,
            valid_actions=valid_actions,
            spatial_encoder=spatial_encoder,
            slot_count=slot_count,
            **memory_kwargs,
        )
    if model_name == "gru":
        return GRUActorCritic(
            action_dim=action_dim,
            d_model=d_model,
            n_layers=getattr(cfg, "gru_layers", getattr(cfg, "lstm_layers", 1)),
            spatial_layers=spatial_layers,
            spatial_heads=spatial_heads,
            dropout=dropout,
            valid_actions=valid_actions,
            spatial_encoder=spatial_encoder,
            slot_count=slot_count,
            **memory_kwargs,
        )
    if model_name == "mamba":
        return MambaActorCritic(
            action_dim=action_dim,
            d_model=d_model,
            n_layers=getattr(cfg, "mamba_layers", 2),
            d_state=getattr(cfg, "d_state", 16),
            d_conv=getattr(cfg, "d_conv", 4),
            expand=getattr(cfg, "expand", 2),
            variant=getattr(cfg, "mamba_variant", "mamba"),
            headdim=getattr(cfg, "mamba_headdim", 64),
            ngroups=getattr(cfg, "mamba_ngroups", 1),
            chunk_size=getattr(cfg, "mamba_chunk_size", 64),
            rope_fraction=getattr(cfg, "mamba_rope_fraction", 0.5),
            spatial_layers=spatial_layers,
            spatial_heads=spatial_heads,
            dropout=dropout,
            valid_actions=valid_actions,
            spatial_encoder=spatial_encoder,
            slot_count=slot_count,
            **memory_kwargs,
        )
    if model_name == "attention":
        return AttentionActorCritic(
            action_dim=action_dim,
            d_model=d_model,
            n_layers=getattr(cfg, "attention_layers", 2),
            n_heads=getattr(cfg, "attention_heads", 4),
            context_len=getattr(cfg, "context_len", 64),
            spatial_layers=spatial_layers,
            spatial_heads=spatial_heads,
            dropout=dropout,
            valid_actions=valid_actions,
            spatial_encoder=spatial_encoder,
            slot_count=slot_count,
            **memory_kwargs,
        )
    if model_name == "gated_attention":
        return FastGatedAttentionActorCritic(
            action_dim=action_dim,
            d_model=d_model,
            n_layers=getattr(cfg, "attention_layers", 2),
            n_heads=getattr(cfg, "attention_heads", 4),
            context_len=getattr(cfg, "context_len", 128),
            spatial_layers=spatial_layers,
            spatial_heads=spatial_heads,
            dropout=dropout,
            valid_actions=valid_actions,
            spatial_encoder=spatial_encoder,
            position_mode=getattr(cfg, "gated_attention_pos", "learned"),
            slot_count=slot_count,
            **memory_kwargs,
        )
    

    raise ValueError(f"Unknown model type: {model_name}")


class _DictConfig:
    def __init__(self, values: dict[str, Any]):
        self.__dict__.update(values)


def _resolve_mamba_block(variant: str):
    if variant == "mamba":
        return MambaBlock
    if variant == "mamba2":
        return Mamba2Block
    if variant == "mamba3":
        return Mamba3Block
    raise ValueError(f"Unknown Mamba variant: {variant}")


def _build_mamba_block(
    *,
    variant: str,
    block_cls,
    d_model: int,
    d_state: int,
    d_conv: int,
    expand: int,
    headdim: int,
    ngroups: int,
    chunk_size: int,
    rope_fraction: float,
    dropout: float,
    layer_idx: int,
):
    if variant == "mamba":
        return block_cls(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            layer_idx=layer_idx,
        )
    if variant == "mamba2":
        return block_cls(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            chunk_size=chunk_size,
            layer_idx=layer_idx,
        )
    if variant == "mamba3":
        return block_cls(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            rope_fraction=rope_fraction,
            chunk_size=chunk_size,
            dropout=dropout,
            layer_idx=layer_idx,
        )
    raise ValueError(f"Unknown Mamba variant: {variant}")


def _raise_if_nonfinite(tensor: torch.Tensor, name: str) -> None:
    if hasattr(torch, "compiler") and torch.compiler.is_compiling():
        return
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} contains NaN or Inf.")


def _build_alibi_slopes(n_heads: int) -> torch.Tensor:
    def slopes_power_of_two(power_heads: int) -> list[float]:
        start = 2.0 ** (-2.0 ** -(math.log2(power_heads) - 3.0))
        ratio = start
        return [start * ratio**i for i in range(power_heads)]

    if n_heads <= 0:
        raise ValueError("n_heads must be positive.")
    if math.log2(n_heads).is_integer():
        slopes = slopes_power_of_two(n_heads)
    else:
        closest_power = 2 ** math.floor(math.log2(n_heads))
        slopes = slopes_power_of_two(closest_power)
        slopes += slopes_power_of_two(2 * closest_power)[0::2][: n_heads - closest_power]
    return torch.tensor(slopes, dtype=torch.float32)


def _alibi_attention_mask(
    seq_len: int,
    slopes: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    *,
    causal: bool,
) -> torch.Tensor:
    positions = torch.arange(seq_len, device=device)
    age = (positions[:, None] - positions[None, :]).clamp_min(0)
    bias = -slopes.to(device=device, dtype=dtype).view(1, -1, 1, 1) * age.to(dtype).view(1, 1, seq_len, seq_len)
    if causal:
        future = positions[None, :] > positions[:, None]
        bias = bias.masked_fill(future.view(1, 1, seq_len, seq_len), torch.finfo(dtype).min)
    return bias


def _apply_valid_mask(x: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    if valid_mask is None:
        return x
    return x * valid_mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)


def _resolve_lengths(
    valid_mask: torch.Tensor | None,
    lengths: torch.Tensor | None,
    seq_len: int,
) -> torch.Tensor | None:
    if lengths is not None:
        return lengths.to(dtype=torch.long).clamp(min=1, max=seq_len)
    if valid_mask is None:
        return None
    return valid_mask.to(dtype=torch.long).sum(dim=1).clamp(min=1, max=seq_len)


def _gather_last_valid(x: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    if valid_mask is None:
        return x[:, -1]
    indices = valid_mask.to(device=x.device).long().sum(dim=1).clamp(min=1, max=x.shape[1]) - 1
    rows = torch.arange(x.shape[0], device=x.device)
    return x[rows, indices]


def _causal_attention_bias(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    positions = torch.arange(seq_len, device=device)
    future = positions[None, :] > positions[:, None]
    bias = torch.zeros((1, 1, seq_len, seq_len), device=device, dtype=dtype)
    return bias.masked_fill(future.view(1, 1, seq_len, seq_len), torch.finfo(dtype).min)


def _padding_attention_bias(valid_mask: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    valid = valid_mask.to(device=device).bool()
    query_valid = valid[:, None, :, None]
    key_invalid = ~valid[:, None, None, :]
    bias = torch.zeros((valid.shape[0], 1, valid.shape[1], valid.shape[1]), device=device, dtype=dtype)
    return bias.masked_fill(query_valid & key_invalid, torch.finfo(dtype).min)


def _temporal_attention_mask(
    valid_mask: torch.Tensor | None,
    seq_len: int,
    n_heads: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if valid_mask is None:
        return None
    valid = valid_mask.to(device=device).bool()
    positions = torch.arange(seq_len, device=device)
    causal = positions[None, :] > positions[:, None]
    padding = valid[:, :, None] & ~valid[:, None, :]
    mask = causal.unsqueeze(0) | padding
    return mask[:, None].expand(-1, n_heads, -1, -1).reshape(-1, seq_len, seq_len)


def _parse_valid_actions(value: Any) -> list[int] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return [int(part) for part in value.split(",") if part.strip()]
    return [int(action) for action in value]


def _register_action_mask(module: nn.Module, action_dim: int, valid_actions: list[int] | None) -> None:
    mask = torch.zeros(action_dim, dtype=torch.float32)
    if valid_actions is not None:
        if not valid_actions:
            raise ValueError("valid_actions cannot be empty.")
        if min(valid_actions) < 0 or max(valid_actions) >= action_dim:
            raise ValueError(f"valid_actions={valid_actions} is outside action_dim={action_dim}.")
        mask.fill_(-1.0e9)
        mask[torch.as_tensor(valid_actions, dtype=torch.long)] = 0.0
    module.register_buffer("action_logit_mask", mask, persistent=False)


def _mask_logits(module: nn.Module, logits: torch.Tensor) -> torch.Tensor:
    return logits + module.action_logit_mask.to(device=logits.device, dtype=logits.dtype)


def _direction_index(direction: torch.Tensor | None, batch: int, device: torch.device) -> torch.Tensor:
    if direction is None:
        return torch.zeros(batch, dtype=torch.long, device=device)
    return direction.long().reshape(batch, -1)[:, 0].clamp_(0, 3)


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.ones((seq_len, seq_len), dtype=torch.bool, device=device).triu_(1)
