"""Evaluate trained MiniGrid Memory checkpoints."""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from collections import deque
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False.*",
    category=UserWarning,
)

from src.envs import make_env
from src.models import _safe_categorical, build_actor_critic


class Config(SimpleNamespace):
    """Compatibility shim for old checkpoints saved with a pickled Config."""


def evaluate(
    checkpoint_path: str,
    episodes: int = 100,
    seed: int = 999,
    env_id: str | None = None,
    deterministic: bool = True,
    allow_legacy_load: bool = False,
) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = _checkpoint_config(ckpt)
    if env_id is not None:
        cfg.env_id = env_id

    probe_env = make_env(cfg.env_id)
    action_dim = getattr(cfg, "action_dim", probe_env.action_space.n)
    probe_env.close()

    model = build_actor_critic(cfg, action_dim=action_dim).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=not allow_legacy_load)
    if allow_legacy_load and (missing or unexpected):
        print(f"Checkpoint loaded with missing={missing} unexpected={unexpected}")
    model.eval()

    successes = 0
    episode_returns = []
    episode_lengths = []

    for episode in range(episodes):
        env = make_env(cfg.env_id, seed=seed + episode)
        obs_dict, _ = env.reset(seed=seed + episode)
        obs = obs_dict["obs"]
        direction = obs_dict["direction"]
        prev_action = obs_dict["prev_action"]
        prev_reward = obs_dict["prev_reward"]
        episode_start = obs_dict["episode_start"]

        obs_ctx = deque(maxlen=cfg.context_len)
        dir_ctx = deque(maxlen=cfg.context_len)
        act_ctx = deque(maxlen=cfg.context_len)
        rew_ctx = deque(maxlen=cfg.context_len)
        start_ctx = deque(maxlen=cfg.context_len)

        done = False
        ep_return = 0.0
        ep_len = 0
        last_reward = 0.0

        while not done:
            if cfg.model in {"mamba", "lstm", "gru", "attention", "gated_attention"}:
                obs_ctx.append(obs)
                dir_ctx.append(direction)
                act_ctx.append(prev_action)
                rew_ctx.append(prev_reward)
                start_ctx.append(episode_start)
                action = _sequence_action(
                    model,
                    device,
                    obs_ctx,
                    dir_ctx,
                    act_ctx,
                    rew_ctx,
                    start_ctx,
                    cfg.context_len,
                    action_dim,
                    deterministic=deterministic,
                )
            else:
                with torch.no_grad():
                    logits, _ = model.forward(
                        torch.as_tensor(obs, device=device).unsqueeze(0),
                        torch.as_tensor(direction, device=device).unsqueeze(0),
                        torch.as_tensor(prev_action, device=device).unsqueeze(0),
                        torch.as_tensor(prev_reward, device=device).unsqueeze(0),
                        torch.as_tensor(episode_start, device=device).unsqueeze(0),
                    )
                    if deterministic:
                        safe_logits = torch.nan_to_num(logits, nan=-1e8, posinf=1e8, neginf=-1e8)
                        action = torch.argmax(safe_logits, dim=-1).item()
                    else:
                        action = _safe_categorical(logits).sample().item()

            obs_dict, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            last_reward = reward
            ep_return += reward
            ep_len += 1

            obs = obs_dict["obs"]
            direction = obs_dict["direction"]
            prev_action = obs_dict["prev_action"]
            prev_reward = obs_dict["prev_reward"]
            episode_start = obs_dict["episode_start"]

        successes += int(last_reward > 0)
        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)
        env.close()

    results = {
        "success_rate": successes / episodes,
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "mean_length": float(np.mean(episode_lengths)),
    }

    print(f"Checkpoint:       {checkpoint_path}")
    print(f"Environment:      {cfg.env_id}")
    print(f"Model:            {cfg.model}")
    print(f"Episodes:         {episodes}")
    print(f"Policy:           {'greedy' if deterministic else 'stochastic'}")
    print(f"Success rate:     {results['success_rate']:.2%} ({successes}/{episodes})")
    print(f"Mean return:      {results['mean_return']:.3f}")
    print(f"Return std:       {results['std_return']:.3f}")
    print(f"Mean length:      {results['mean_length']:.1f}")
    return results


def _sequence_action(
    model,
    device: torch.device,
    obs_ctx: deque,
    dir_ctx: deque,
    act_ctx: deque,
    rew_ctx: deque,
    start_ctx: deque,
    context_len: int,
    action_dim: int,
    *,
    deterministic: bool,
) -> int:
    obs_seq, dir_seq, act_seq, rew_seq, start_seq, valid_mask = _pack_context(
        obs_ctx,
        dir_ctx,
        act_ctx,
        rew_ctx,
        start_ctx,
        context_len,
        action_dim,
    )
    with torch.no_grad():
        logits, _ = model.forward(
            torch.as_tensor(obs_seq, device=device),
            torch.as_tensor(dir_seq, device=device),
            torch.as_tensor(act_seq, device=device),
            torch.as_tensor(rew_seq, device=device),
            torch.as_tensor(start_seq, device=device),
            valid_mask=torch.as_tensor(valid_mask, device=device),
        )
        mask_t = torch.as_tensor(valid_mask, device=device)
        last_idx = mask_t.long().sum(dim=1).clamp(min=1, max=logits.shape[1]) - 1
        last_logits = logits[torch.arange(logits.shape[0], device=device), last_idx]
        if deterministic:
            safe_logits = torch.nan_to_num(last_logits, nan=-1e8, posinf=1e8, neginf=-1e8)
            return torch.argmax(safe_logits, dim=-1).item()
        return _safe_categorical(last_logits).sample().item()


def _pack_context(
    obs_ctx: deque,
    dir_ctx: deque,
    act_ctx: deque,
    rew_ctx: deque,
    start_ctx: deque,
    context_len: int,
    action_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(obs_ctx)
    obs_shape = np.asarray(obs_ctx[0]).shape if n else (7, 7, 3)
    obs_seq = np.zeros((1, context_len, *obs_shape), dtype=np.uint8)
    dir_seq = np.zeros((1, context_len, 1), dtype=np.int64)
    act_seq = np.zeros((1, context_len, action_dim), dtype=np.float32)
    rew_seq = np.zeros((1, context_len, 1), dtype=np.float32)
    start_seq = np.zeros((1, context_len, 1), dtype=np.float32)
    valid_mask = np.zeros((1, context_len), dtype=np.float32)

    if n:
        obs_seq[0, :n] = np.asarray(list(obs_ctx), dtype=np.uint8)
        dir_seq[0, :n] = np.asarray(list(dir_ctx), dtype=np.int64)
        act_seq[0, :n] = np.asarray(list(act_ctx), dtype=np.float32)
        rew_seq[0, :n] = np.asarray(list(rew_ctx), dtype=np.float32)
        start_seq[0, :n] = np.asarray(list(start_ctx), dtype=np.float32)
        valid_mask[0, :n] = 1.0

    return obs_seq, dir_seq, act_seq, rew_seq, start_seq, valid_mask


def _checkpoint_config(ckpt) -> SimpleNamespace:
    cfg_raw = ckpt.get("config_dict", ckpt.get("config"))
    if isinstance(cfg_raw, dict):
        values = dict(cfg_raw)
    else:
        values = dict(getattr(cfg_raw, "__dict__", {}))

    values.setdefault("model", "lstm")
    values.setdefault("env_id", "MiniGrid-MemoryS13-v0")
    values.setdefault("context_len", 128)
    values.setdefault("d_model", 128)
    values.setdefault("spatial_encoder", "hybrid")
    values.setdefault("spatial_layers", 2)
    values.setdefault("spatial_heads", 4)
    values.setdefault("dropout", 0.0)
    values.setdefault("lstm_layers", 1)
    values.setdefault("gru_layers", 1)
    values.setdefault("slot_count", 0)
    values.setdefault("slot_extractor", "query_pool")
    values.setdefault("slot_iters", 3)
    values.setdefault("slot_mlp_ratio", 2.0)
    values.setdefault("temporal_token_mode", "flatten")
    values.setdefault("memory_kind", "none")
    values.setdefault("memory_slots", 16)
    values.setdefault("memory_topk", 4)
    values.setdefault("memory_write_window", 12)
    values.setdefault("aux_recall_coef", 0.0)
    values.setdefault("mamba_variant", "mamba")
    values.setdefault("mamba_layers", 2)
    values.setdefault("d_state", 32)
    values.setdefault("d_conv", 4)
    values.setdefault("expand", 2)
    values.setdefault("mamba_headdim", 64)
    values.setdefault("mamba_ngroups", 1)
    values.setdefault("mamba_chunk_size", 64)
    values.setdefault("mamba_rope_fraction", 0.5)
    values.setdefault("attention_layers", 2)
    values.setdefault("attention_heads", 4)
    values.setdefault("gated_attention_pos", "learned")
    values.setdefault("valid_actions", "0,1,2")
    return SimpleNamespace(**values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--env-id", type=str, default=None, help="Override checkpoint env")
    parser.add_argument("--stochastic", action="store_true", help="Sample instead of greedy actions")
    parser.add_argument("--allow-legacy-load", action="store_true", help="Load old checkpoints with strict=False")
    args = parser.parse_args()

    evaluate(
        args.checkpoint,
        episodes=args.episodes,
        seed=args.seed,
        env_id=args.env_id,
        deterministic=not args.stochastic,
        allow_legacy_load=args.allow_legacy_load,
    )
