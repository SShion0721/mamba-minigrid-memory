"""Render trained agents and optionally save MP4 videos."""

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


ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]


class Config(SimpleNamespace):
    """Compatibility shim for old checkpoints saved with a pickled Config."""


def record_episodes(
    checkpoint_path: str,
    num_episodes: int = 3,
    seed: int = 42,
    save_dir: str = "videos",
    fps: int = 8,
    deterministic: bool = True,
    allow_legacy_load: bool = False,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = _checkpoint_config(ckpt)

    probe_env = make_env(cfg.env_id)
    action_dim = getattr(cfg, "action_dim", probe_env.action_space.n)
    probe_env.close()

    model = build_actor_critic(cfg, action_dim=action_dim).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=not allow_legacy_load)
    if allow_legacy_load and (missing or unexpected):
        print(f"Checkpoint loaded with missing={missing} unexpected={unexpected}")
    model.eval()

    os.makedirs(save_dir, exist_ok=True)
    imageio = _try_imageio()

    for episode in range(num_episodes):
        env = make_env(cfg.env_id, seed=seed + episode, render_mode="rgb_array")
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

        frames = []
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        done = False
        ep_return = 0.0
        step_count = 0
        last_reward = 0.0

        print(f"\nEpisode {episode + 1}/{num_episodes}")

        while not done:
            if cfg.model in {"mamba", "lstm", "gru", "attention", "gated_attention"}:
                obs_ctx.append(obs)
                dir_ctx.append(direction)
                act_ctx.append(prev_action)
                rew_ctx.append(prev_reward)
                start_ctx.append(episode_start)
                action, probs = _sequence_action(
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
                    safe_logits = torch.nan_to_num(logits, nan=-1e8, posinf=1e8, neginf=-1e8)
                    probs = torch.softmax(safe_logits, dim=-1).cpu().numpy()[0]
                    if deterministic:
                        action = torch.argmax(safe_logits, dim=-1).item()
                    else:
                        action = _safe_categorical(logits).sample().item()

            obs_dict, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            last_reward = reward
            ep_return += reward
            step_count += 1

            frame = env.render()
            if frame is not None:
                frames.append(frame)

            obs = obs_dict["obs"]
            direction = obs_dict["direction"]
            prev_action = obs_dict["prev_action"]
            prev_reward = obs_dict["prev_reward"]
            episode_start = obs_dict["episode_start"]

            action_name = ACTION_NAMES[action] if action < len(ACTION_NAMES) else str(action)
            prob_text = ", ".join(f"{p:.2f}" for p in probs)
            print(f"  step {step_count:>3d}: {action_name:<7s} probs=[{prob_text}] return={ep_return:.3f}")

        status = "SUCCESS" if last_reward > 0 else "FAIL"
        print(f"  result: {status} | return={ep_return:.3f} | steps={step_count}")

        if imageio is not None and frames:
            out_path = os.path.join(save_dir, f"{cfg.model}_{cfg.env_id}_ep{episode + 1}.mp4")
            imageio.mimsave(out_path, frames, fps=fps)
            print(f"  video: {out_path}")
        elif frames:
            print("  video skipped: install imageio[ffmpeg] to save MP4 output")

        env.close()


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
) -> tuple[int, np.ndarray]:
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
        safe_logits = torch.nan_to_num(last_logits, nan=-1e8, posinf=1e8, neginf=-1e8)
        probs = torch.softmax(safe_logits, dim=-1).cpu().numpy()[0]
        if deterministic:
            action = torch.argmax(safe_logits, dim=-1).item()
        else:
            action = _safe_categorical(last_logits).sample().item()
    return action, probs


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


def _try_imageio():
    try:
        import imageio.v2 as imageio
    except Exception:
        return None
    return imageio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="videos")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--stochastic", action="store_true", help="Sample instead of greedy actions")
    parser.add_argument("--allow-legacy-load", action="store_true", help="Load old checkpoints with strict=False")
    args = parser.parse_args()

    record_episodes(
        args.checkpoint,
        num_episodes=args.episodes,
        seed=args.seed,
        save_dir=args.save_dir,
        fps=args.fps,
        deterministic=not args.stochastic,
        allow_legacy_load=args.allow_legacy_load,
    )
