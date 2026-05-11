"""Actor-critic networks for MiniGrid PPO experiments."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

try:
    from mamba_ssm import Mamba as MambaBlock
    try:
        from mamba_ssm import Mamba2 as Mamba2Block
    except Exception:
        Mamba2Block = None
    try:
        from mamba_ssm import Mamba3 as Mamba3Block
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


class MiniGridSpatialEncoder(nn.Module):
    """Encode a 7x7 semantic MiniGrid view with lightweight spatial attention."""

    def __init__(
        self,
        d_model: int = 128,
        spatial_layers: int = 2,
        spatial_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.spatial_layers = spatial_layers

        self.obj_emb = nn.Embedding(32, 24)
        self.color_emb = nn.Embedding(16, 16)
        self.state_emb = nn.Embedding(8, 8)
        self.cell_proj = layer_init(nn.Linear(24 + 16 + 8, d_model))

        self.direction_emb = nn.Embedding(4, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, 49, d_model))

        if spatial_layers > 0:
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
        else:
            self.spatial = None
            self.flat = nn.Sequential(
                layer_init(nn.Linear(49 * d_model + d_model, 2 * d_model)),
                nn.GELU(),
                layer_init(nn.Linear(2 * d_model, d_model)),
                nn.GELU(),
            )

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
            out = self.flat(torch.cat([cells.reshape(num_items, -1), direction_token.squeeze(1)], dim=-1))
        else:
            tokens = torch.cat([cls, cells], dim=1)
            out = self.out_norm(self.spatial(tokens)[:, 0])

        if has_time:
            out = out.reshape(batch, seq_len, self.d_model)
        return out


class TokenEncoder(nn.Module):
    """Build trajectory tokens from MiniGrid frame, direction, and side inputs."""

    def __init__(
        self,
        action_dim: int,
        d_model: int,
        spatial_layers: int = 2,
        spatial_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.obs_encoder = MiniGridSpatialEncoder(
            d_model=d_model,
            spatial_layers=spatial_layers,
            spatial_heads=spatial_heads,
            dropout=dropout,
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
        x = self.obs_encoder(obs_seq, direction_seq)
        x = (
            x
            + self.action_proj(prev_action_seq.float())
            + self.reward_proj(prev_reward_seq.float())
            + self.start_proj(episode_start_seq.float())
        )
        return self.token_norm(x)


class MLPActorCritic(nn.Module):
    """Feedforward PPO baseline with spatial attention but no temporal memory."""

    def __init__(
        self,
        action_dim: int = 7,
        d_model: int = 128,
        spatial_layers: int = 2,
        spatial_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.encoder = MiniGridSpatialEncoder(
            d_model=d_model,
            spatial_layers=spatial_layers,
            spatial_heads=spatial_heads,
            dropout=dropout,
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
        return self.actor(x), self.critic(x).squeeze(-1)

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
        dist = Categorical(logits=logits)
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
    ):
        super().__init__()
        self.action_dim = action_dim
        self.token_encoder = TokenEncoder(
            action_dim=action_dim,
            d_model=d_model,
            spatial_layers=spatial_layers,
            spatial_heads=spatial_heads,
            dropout=dropout,
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
        return self.actor(x), self.critic(x).squeeze(-1)

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
        dist = Categorical(logits=logits[:, -1])
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
        spatial_layers: int = 2,
        spatial_heads: int = 4,
        dropout: float = 0.0,
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
        self.token_encoder = TokenEncoder(
            action_dim=action_dim,
            d_model=d_model,
            spatial_layers=spatial_layers,
            spatial_heads=spatial_heads,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList(
            [
                block_cls(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(n_layers)
            ]
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
        for block in self.blocks:
            x = x + block(x)
        x = self.norm(x)
        return self.actor(x), self.critic(x).squeeze(-1)

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
        dist = Categorical(logits=logits[:, -1])
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), values[:, -1]


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
    ):
        super().__init__()
        self.action_dim = action_dim
        self.context_len = context_len
        self.token_encoder = TokenEncoder(
            action_dim=action_dim,
            d_model=d_model,
            spatial_layers=spatial_layers,
            spatial_heads=spatial_heads,
            dropout=dropout,
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
        return self.actor(x), self.critic(x).squeeze(-1)

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
        dist = Categorical(logits=logits[:, -1])
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
    dropout = getattr(cfg, "dropout", 0.0)

    if model_name == "mlp":
        return MLPActorCritic(
            action_dim=action_dim,
            d_model=d_model,
            spatial_layers=spatial_layers,
            spatial_heads=spatial_heads,
            dropout=dropout,
        )
    if model_name == "lstm":
        return LSTMActorCritic(
            action_dim=action_dim,
            d_model=d_model,
            n_layers=getattr(cfg, "lstm_layers", 1),
            spatial_layers=spatial_layers,
            spatial_heads=spatial_heads,
            dropout=dropout,
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
            spatial_layers=spatial_layers,
            spatial_heads=spatial_heads,
            dropout=dropout,
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


def _direction_index(direction: torch.Tensor | None, batch: int, device: torch.device) -> torch.Tensor:
    if direction is None:
        return torch.zeros(batch, dtype=torch.long, device=device)
    return direction.long().reshape(batch, -1)[:, 0].clamp_(0, 3)


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.full((seq_len, seq_len), float("-inf"), device=device).triu_(1)
