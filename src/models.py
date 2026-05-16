"""Actor-critic networks for MiniGrid PPO experiments."""

from __future__ import annotations

from typing import Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

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
    ):
        super().__init__()
        self.d_model = d_model
        self.spatial_layers = spatial_layers
        self.encoder_type = encoder_type

        self.obj_emb = nn.Embedding(32, 24)
        self.color_emb = nn.Embedding(16, 16)
        self.state_emb = nn.Embedding(8, 8)
        self.cell_proj = layer_init(nn.Linear(24 + 16 + 8, d_model))

        self.direction_emb = nn.Embedding(4, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, 49, d_model))

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

    def forward(self, obs: torch.Tensor, direction: torch.Tensor | None = None) -> torch.Tensor:
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
            else:
                grid = cells.transpose(1, 2).reshape(num_items, self.d_model, 7, 7)
                for block in self.conv_blocks:
                    grid = block(grid)
                tokens = grid.flatten(2).transpose(1, 2)
                weights = torch.softmax(self.saliency(tokens).squeeze(-1), dim=-1).unsqueeze(-1)
                salient = (tokens * weights).sum(dim=1)
                mean = tokens.mean(dim=1)
                out = self.out_norm(self.summary(torch.cat([salient, mean, direction_token.squeeze(1)], dim=-1)))
        else:
            tokens = torch.cat([cls, cells], dim=1)
            out = self.out_norm(self.spatial(tokens)[:, 0])

        if has_time:
            out = out.reshape(batch, seq_len, self.d_model)
        return out


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
    ):
        super().__init__()
        self.obs_encoder = MiniGridSpatialEncoder(
            d_model=d_model,
            spatial_layers=spatial_layers,
            spatial_heads=spatial_heads,
            dropout=dropout,
            encoder_type=spatial_encoder,
        )
        self.action_proj = layer_init(nn.Linear(action_dim, d_model))
        self.reward_proj = layer_init(nn.Linear(1, d_model))
        self.start_proj = layer_init(nn.Linear(1, d_model))
        self.token_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        obs_seq: torch.Tensor,
        direction_seq: torch.Tensor,
        prev_action_seq: torch.Tensor,
        prev_reward_seq: torch.Tensor,
        episode_start_seq: torch.Tensor,
    ) -> torch.Tensor:
        x = (
            self.obs_encoder(obs_seq, direction_seq)
            + self.action_proj(prev_action_seq.float())
            + self.reward_proj(prev_reward_seq.float())
            + self.start_proj(episode_start_seq.float())
        )
        return self.token_norm(x)


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
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        attn_out = self.out_proj(attn_out)
        
        # 使用 GRU 门控替代原来的简单残差 x = x + attn_out
        x = self.gate1(x, attn_out)
        
        # 2. MLP 层
        mlp_out = self.mlp(self.norm2(x))
        x = self.gate2(x, mlp_out)
        
        return x

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

    def forward(
        self,
        obs: torch.Tensor,
        direction: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        episode_start: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = (
            self.encoder(obs, direction)
            + self.action_proj(prev_action.float())
            + self.reward_proj(prev_reward.float())
            + self.start_proj(episode_start.float())
        )
        x = self.shared(x)
        return _mask_logits(self, self.actor(x)), self.critic(x).squeeze(-1)

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

    def forward(
        self,
        obs_seq: torch.Tensor,
        direction_seq: torch.Tensor,
        prev_action_seq: torch.Tensor,
        prev_reward_seq: torch.Tensor,
        episode_start_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.token_encoder(obs_seq, direction_seq, prev_action_seq, prev_reward_seq, episode_start_seq)
        x, _ = self.lstm(x)
        x = self.norm(x)
        return _mask_logits(self, self.actor(x)), self.critic(x).squeeze(-1)

    def get_action_and_value(
        self,
        obs_seq: torch.Tensor,
        direction_seq: torch.Tensor,
        prev_action_seq: torch.Tensor,
        prev_reward_seq: torch.Tensor,
        episode_start_seq: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(obs_seq, direction_seq, prev_action_seq, prev_reward_seq, episode_start_seq)
        dist = _safe_categorical(logits[:, -1])
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), values[:, -1]


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

        #增加mamba梯度稳定性：在每个block输出后添加残差连接和LayerNorm，并在输出前对block输出进行clamp，防止数值过大导致NaN。

    def forward(
        self,
        obs_seq: torch.Tensor,
        direction_seq: torch.Tensor,
        prev_action_seq: torch.Tensor,
        prev_reward_seq: torch.Tensor,
        episode_start_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.token_encoder(obs_seq, direction_seq, prev_action_seq, prev_reward_seq, episode_start_seq)
        for norm, block in zip(self.block_norms, self.blocks):
            out = block(norm(x))
            _raise_if_nonfinite(out, "Mamba block output")
            x = x + self.residual_scale * out
        x = self.norm(x)
        return _mask_logits(self, self.actor(x)), self.critic(x).squeeze(-1)

    def init_inference_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> list[tuple[torch.Tensor, ...]]:
        """Allocate Mamba recurrent caches for one-token rollout inference."""

        states: list[tuple[torch.Tensor, ...]] = []
        for block in self.blocks:
            if not hasattr(block, "allocate_inference_cache"):
                raise RuntimeError(f"{type(block).__name__} does not support inference caching.")
            cache = block.allocate_inference_cache(batch_size, 1, dtype=dtype)
            cache = tuple(t.to(device=device) if device is not None else t for t in cache)
            states.append(cache)
        return states

    @staticmethod
    def reset_inference_state(
        inference_state: list[tuple[torch.Tensor, ...]] | None,
        done_mask: torch.Tensor,
    ) -> None:
        if inference_state is None or done_mask.numel() == 0 or not done_mask.any():
            return
        for cache in inference_state:
            for tensor in cache:
                tensor[done_mask] = 0

    def get_action_and_value_step(
        self,
        obs: torch.Tensor,
        direction: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        episode_start: torch.Tensor,
        inference_state: list[tuple[torch.Tensor, ...]],
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, ...]]]:
        x = self.token_encoder(
            obs.unsqueeze(1),
            direction.unsqueeze(1),
            prev_action.unsqueeze(1),
            prev_reward.unsqueeze(1),
            episode_start.unsqueeze(1),
        )
        new_state: list[tuple[torch.Tensor, ...]] = []
        for cache, norm, block in zip(inference_state, self.block_norms, self.blocks):
            out = block.step(norm(x), *cache)
            block_out = out[0]
            block_cache = tuple(out[1:])
            _raise_if_nonfinite(block_out, "Mamba step output")
            x = x + self.residual_scale * block_out
            new_state.append(block_cache)

        x = self.norm(x)
        logits = _mask_logits(self, self.actor(x[:, -1]))
        value = self.critic(x[:, -1]).squeeze(-1)
        dist = _safe_categorical(logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value, new_state

    def get_action_and_value(
        self,
        obs_seq: torch.Tensor,
        direction_seq: torch.Tensor,
        prev_action_seq: torch.Tensor,
        prev_reward_seq: torch.Tensor,
        episode_start_seq: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(obs_seq, direction_seq, prev_action_seq, prev_reward_seq, episode_start_seq)
        dist = _safe_categorical(logits[:, -1])
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), values[:, -1]


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
    ):
        super().__init__()
        if position_mode not in {"learned", "none", "alibi"}:
            raise ValueError("position_mode must be one of: learned, none, alibi.")
        self.action_dim = action_dim
        _register_action_mask(self, action_dim, valid_actions)
        self.context_len = context_len
        self.position_mode = position_mode
        self.n_heads = n_heads
        self.d_model = d_model
        
        self.token_encoder = TokenEncoder(
            action_dim=action_dim, d_model=d_model, spatial_layers=spatial_layers,
            spatial_heads=spatial_heads, dropout=dropout, spatial_encoder=spatial_encoder,
        )
        
        if self.position_mode == "learned":
            self.temporal_pos = nn.Parameter(torch.zeros(1, context_len, d_model))
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

    def forward(
        self, obs_seq, direction_seq, prev_action_seq, prev_reward_seq, episode_start_seq
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        x = self.token_encoder(obs_seq, direction_seq, prev_action_seq, prev_reward_seq, episode_start_seq)
        seq_len = x.shape[1]
        
        if seq_len > self.context_len:
            raise ValueError(f"Sequence length {seq_len} exceeds context_len {self.context_len}.")
            
        if self.temporal_pos is not None:
            x = x + self.temporal_pos[:, -seq_len:]
        
        # 前向传播过 Gated Blocks
        for block in self.blocks:
            x = block(x, is_causal=True, alibi_slopes=self.alibi_slopes)
            
        x = self.norm(x)
        return _mask_logits(self, self.actor(x)), self.critic(x).squeeze(-1)

    def get_action_and_value(self, obs_seq, direction_seq, prev_action_seq, prev_reward_seq, episode_start_seq, action=None):
        logits, values = self.forward(obs_seq, direction_seq, prev_action_seq, prev_reward_seq, episode_start_seq)
        dist = _safe_categorical(logits[:, -1])
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), values[:, -1]

    def init_inference_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
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
        return states

    @staticmethod
    def reset_inference_state(
        inference_state: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None,
        done_mask: torch.Tensor,
    ) -> None:
        if inference_state is None or done_mask.numel() == 0 or not done_mask.any():
            return
        for cache_k, cache_v, lengths in inference_state:
            cache_k[done_mask] = 0
            cache_v[done_mask] = 0
            lengths[done_mask] = 0

    def get_action_and_value_step(
        self,
        obs: torch.Tensor,
        direction: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: torch.Tensor,
        episode_start: torch.Tensor,
        inference_state: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        x = self.token_encoder(
            obs.unsqueeze(1),
            direction.unsqueeze(1),
            prev_action.unsqueeze(1),
            prev_reward.unsqueeze(1),
            episode_start.unsqueeze(1),
        )
        new_state = []
        for block, cache in zip(self.blocks, inference_state):
            x, cache = block.forward_step(x, cache, alibi_slopes=self.alibi_slopes)
            new_state.append(cache)

        x = self.norm(x)
        logits = _mask_logits(self, self.actor(x[:, -1]))
        value = self.critic(x[:, -1]).squeeze(-1)
        dist = _safe_categorical(logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value, new_state
    
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
    ):
        super().__init__()
        self.action_dim = action_dim
        _register_action_mask(self, action_dim, valid_actions)
        self.context_len = context_len
        self.token_encoder = TokenEncoder(
            action_dim=action_dim,
            d_model=d_model,
            spatial_layers=spatial_layers,
            spatial_heads=spatial_heads,
            dropout=dropout,
            spatial_encoder=spatial_encoder,
        )
        self.temporal_pos = nn.Parameter(torch.zeros(1, context_len, d_model))
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
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

    def forward(
        self,
        obs_seq: torch.Tensor,
        direction_seq: torch.Tensor,
        prev_action_seq: torch.Tensor,
        prev_reward_seq: torch.Tensor,
        episode_start_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.token_encoder(obs_seq, direction_seq, prev_action_seq, prev_reward_seq, episode_start_seq)
        seq_len = x.shape[1]
        if seq_len > self.context_len:
            raise ValueError(f"Sequence length {seq_len} exceeds context_len {self.context_len}.")
        x = x + self.temporal_pos[:, -seq_len:]
        mask = _causal_mask(seq_len, x.device)
        x = self.temporal(x, mask=mask)
        x = self.norm(x)
        return _mask_logits(self, self.actor(x)), self.critic(x).squeeze(-1)

    def get_action_and_value(
        self,
        obs_seq: torch.Tensor,
        direction_seq: torch.Tensor,
        prev_action_seq: torch.Tensor,
        prev_reward_seq: torch.Tensor,
        episode_start_seq: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(obs_seq, direction_seq, prev_action_seq, prev_reward_seq, episode_start_seq)
        dist = _safe_categorical(logits[:, -1])
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), values[:, -1]


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

    if model_name == "mlp":
        return MLPActorCritic(
            action_dim=action_dim,
            d_model=d_model,
            spatial_layers=spatial_layers,
            spatial_heads=spatial_heads,
            dropout=dropout,
            valid_actions=valid_actions,
            spatial_encoder=spatial_encoder,
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
    return torch.full((seq_len, seq_len), float("-inf"), device=device).triu_(1)
