"""Train PPO agents for MiniGrid Memory tasks.

Examples:
  python src/train_mamba_ppo.py --model mamba --env-id MiniGrid-MemoryS11-v0
  python src/train_mamba_ppo.py --model attention --env-id MiniGrid-MemoryS11-v0
  python src/train_mamba_ppo.py --model lstm --env-id MiniGrid-MemoryS11-v0
  python src/train_mamba_ppo.py --model mlp --env-id MiniGrid-MemoryS11-v0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from collections import deque
from dataclasses import asdict, dataclass
from typing import Callable
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False.*",
    category=UserWarning,
)

from src.envs import MEMORY_ENVS, make_env
from src.models import build_actor_critic
from src.ppo import PPOTrainer, RolloutBuffer


@dataclass
class Config:
    env_id: str = "MiniGrid-MemoryS11-v0"
    model: str = "mamba"
    seed: int = 42

    total_steps: int = 1_000_000
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    ent_coef_final: float = 0.001

    num_envs: int = 16
    num_steps: int = 128
    batch_size: int = 256
    n_epochs: int = 4

    context_len: int = 64
    chunk_len: int = 64
    batch_chunks: int = 8

    d_model: int = 128
    spatial_encoder: str = "hybrid"
    spatial_layers: int = 2
    spatial_heads: int = 4
    dropout: float = 0.0
    lstm_layers: int = 1
    mamba_variant: str = "mamba"
    mamba_layers: int = 2
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    attention_layers: int = 2
    attention_heads: int = 4
    valid_actions: str = "0,1,2"

    eval_interval: int = 20_000
    eval_episodes: int = 30
    save_interval: int = 100_000
    log_interval: int = 10
    progress_bar: bool = True
    run_name: str = ""
    resume_from: str = ""


def train(config: Config) -> None:
    if config.env_id not in MEMORY_ENVS and "Memory" in config.env_id:
        print(f"Warning: {config.env_id} is not in the known Memory env list.")
    if config.model in {"mamba", "lstm", "attention"} and config.chunk_len > config.num_steps:
        raise ValueError("--chunk-len must be <= --num-steps for sequence models.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    run_name = config.run_name or _default_run_name(config)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_dir = os.path.join(project_root, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    writer = SummaryWriter(run_dir)
    writer.add_text("config", _format_config(config), 0)

    envs = [make_env(config.env_id, seed=config.seed + i) for i in range(config.num_envs)]

    first_obs, _ = envs[0].reset(seed=config.seed)
    obs_shape = first_obs["obs"].shape
    action_dim = envs[0].action_space.n

    model = build_actor_critic(config, action_dim=action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.total_steps // config.batch_size, # 你的总更新次数
        eta_min=1e-10  # 降到最低是多少
    )
    global_step = 0
    if config.resume_from:
        ckpt = torch.load(config.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        global_step = int(ckpt.get("global_step", 0))
        print(f"Resumed checkpoint {config.resume_from} at global_step={global_step}")

    trainer = PPOTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        writer=writer,
        clip_coef=config.clip_coef,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        target_kl=config.target_kl,
    )

    obs_buf = np.zeros((config.num_envs, *obs_shape), dtype=np.float32)
    direction_buf = np.zeros((config.num_envs, 1), dtype=np.int64)
    prev_action_buf = np.zeros((config.num_envs, action_dim), dtype=np.float32)
    prev_reward_buf = np.zeros((config.num_envs, 1), dtype=np.float32)
    episode_start_buf = np.ones((config.num_envs, 1), dtype=np.float32)

    obs_buf[0] = first_obs["obs"]
    direction_buf[0] = first_obs["direction"]
    prev_action_buf[0] = first_obs["prev_action"]
    prev_reward_buf[0] = first_obs["prev_reward"]
    episode_start_buf[0] = first_obs["episode_start"]

    for env_idx in range(1, config.num_envs):
        obs_dict, _ = envs[env_idx].reset(seed=config.seed + env_idx)
        obs_buf[env_idx] = obs_dict["obs"]
        direction_buf[env_idx] = obs_dict["direction"]
        prev_action_buf[env_idx] = obs_dict["prev_action"]
        prev_reward_buf[env_idx] = obs_dict["prev_reward"]
        episode_start_buf[env_idx] = obs_dict["episode_start"]

    buffer = RolloutBuffer(
        num_envs=config.num_envs,
        num_steps=config.num_steps,
        obs_shape=obs_shape,
        action_dim=action_dim,
        context_len=config.context_len,
    )

    episode_returns = np.zeros(config.num_envs, dtype=np.float32)
    episode_lengths = np.zeros(config.num_envs, dtype=np.int32)
    recent_returns = deque(maxlen=100)
    recent_lengths = deque(maxlen=100)
    recent_successes = deque(maxlen=100)

    remaining_steps = max(config.total_steps - global_step, config.num_envs * config.num_steps)
    num_updates = max(1, remaining_steps // (config.num_envs * config.num_steps))
    start_time = time.time()
    initial_global_step = global_step
    next_eval = _next_interval_boundary(global_step, config.eval_interval)
    next_save = _next_interval_boundary(global_step, config.save_interval)

    print(
        "Training config: "
        f"run={run_name} env={config.env_id} model={config.model}/{config.mamba_variant} "
        f"steps={global_step}->{config.total_steps} num_envs={config.num_envs} "
        f"rollout={config.num_steps} context={config.context_len} "
        f"chunk={config.chunk_len} batch_chunks={config.batch_chunks} "
        f"spatial={config.spatial_encoder} valid_actions={config.valid_actions}",
        flush=True,
    )

    with tqdm(
        total=config.total_steps,
        initial=min(global_step, config.total_steps),
        desc=f"{config.model}:{config.env_id}",
        unit="step",
        dynamic_ncols=True,
        mininterval=1.0,
        smoothing=0.05,
        file=sys.stdout,
        disable=not config.progress_bar,
        ascii=True,
    ) as pbar:
        for update in range(num_updates):
            progress = update / max(num_updates, 1)
            trainer.ent_coef = config.ent_coef + progress * (config.ent_coef_final - config.ent_coef)
            if config.anneal_lr:
                frac = 1.0 - progress
                optimizer.param_groups[0]["lr"] = config.learning_rate * frac

            model.train()
            for step in range(config.num_steps):
                pbar.set_postfix_str(
                    _progress_status(
                        phase=f"rollout {step + 1}/{config.num_steps}",
                        global_step=global_step,
                        initial_step=initial_global_step,
                        start_time=start_time,
                        optimizer=optimizer,
                        recent_returns=recent_returns,
                        recent_successes=recent_successes,
                    )
                )
                if config.model in {"mamba", "lstm", "attention"}:
                    context = buffer.get_context(
                        obs_buf,
                        direction_buf,
                        prev_action_buf,
                        prev_reward_buf,
                        episode_start_buf,
                    )
                    with torch.no_grad():
                        action, logprob, _, value = model.get_action_and_value(
                            torch.as_tensor(context[0], device=device),
                            torch.as_tensor(context[1], device=device),
                            torch.as_tensor(context[2], device=device),
                            torch.as_tensor(context[3], device=device),
                            torch.as_tensor(context[4], device=device),
                        )
                else:
                    with torch.no_grad():
                        action, logprob, _, value = model.get_action_and_value(
                            torch.as_tensor(obs_buf, device=device),
                            torch.as_tensor(direction_buf, device=device),
                            torch.as_tensor(prev_action_buf, device=device),
                            torch.as_tensor(prev_reward_buf, device=device),
                            torch.as_tensor(episode_start_buf, device=device),
                        )

                action_np = action.cpu().numpy()
                logprob_np = logprob.cpu().numpy()
                value_np = value.cpu().numpy()

                for env_idx, env in enumerate(envs):
                    current_obs = obs_buf[env_idx].copy()
                    current_direction = direction_buf[env_idx].copy()
                    current_prev_action = prev_action_buf[env_idx].copy()
                    current_prev_reward = prev_reward_buf[env_idx].copy()
                    current_episode_start = episode_start_buf[env_idx].copy()

                    obs_dict, reward, terminated, truncated, _ = env.step(action_np[env_idx])
                    done = terminated or truncated

                    buffer.add(
                        env_idx=env_idx,
                        obs=current_obs,
                        direction=current_direction,
                        action=int(action_np[env_idx]),
                        logprob=float(logprob_np[env_idx]),
                        value=float(value_np[env_idx]),
                        reward=float(reward),
                        done=done,
                        prev_action=current_prev_action,
                        prev_reward=current_prev_reward,
                        episode_start=current_episode_start,
                    )

                    episode_returns[env_idx] += reward
                    episode_lengths[env_idx] += 1

                    if done:
                        writer.add_scalar("charts/episodic_return", episode_returns[env_idx], global_step)
                        writer.add_scalar("charts/episodic_length", episode_lengths[env_idx], global_step)
                        writer.add_scalar("charts/train_success", float(reward > 0), global_step)
                        recent_returns.append(float(episode_returns[env_idx]))
                        recent_lengths.append(int(episode_lengths[env_idx]))
                        recent_successes.append(float(reward > 0))
                        episode_returns[env_idx] = 0.0
                        episode_lengths[env_idx] = 0
                        obs_dict, _ = env.reset()

                    obs_buf[env_idx] = obs_dict["obs"]
                    direction_buf[env_idx] = obs_dict["direction"]
                    prev_action_buf[env_idx] = obs_dict["prev_action"]
                    prev_reward_buf[env_idx] = obs_dict["prev_reward"]
                    episode_start_buf[env_idx] = obs_dict["episode_start"]

                buffer.step += 1
                global_step += config.num_envs
                pbar.update(min(config.num_envs, max(config.total_steps - pbar.n, 0)))

            pbar.set_postfix_str(
                _progress_status(
                    phase="bootstrap",
                    global_step=global_step,
                    initial_step=initial_global_step,
                    start_time=start_time,
                    optimizer=optimizer,
                    recent_returns=recent_returns,
                    recent_successes=recent_successes,
                )
            )
            next_value = _bootstrap_value(
                model,
                config,
                buffer,
                device,
                obs_buf,
                direction_buf,
                prev_action_buf,
                prev_reward_buf,
                episode_start_buf,
            )
            buffer.compute_gae(next_value, gamma=config.gamma, gae_lambda=config.gae_lambda)

            trainer.global_step = global_step
            pbar.set_postfix_str(
                _progress_status(
                    phase="ppo update 0/?",
                    global_step=global_step,
                    initial_step=initial_global_step,
                    start_time=start_time,
                    optimizer=optimizer,
                    recent_returns=recent_returns,
                    recent_successes=recent_successes,
                )
            )
            def _ppo_progress(done_batches: int, total_batches: int) -> None:
                pbar.set_postfix_str(
                    _progress_status(
                        phase=f"ppo update {done_batches}/{total_batches}",
                        global_step=global_step,
                        initial_step=initial_global_step,
                        start_time=start_time,
                        optimizer=optimizer,
                        recent_returns=recent_returns,
                        recent_successes=recent_successes,
                    )
                )
            if config.model in {"mamba", "lstm", "attention"}:
                trainer.train_sequence_with_callback(
                    buffer,
                    chunk_len=config.chunk_len,
                    batch_chunks=config.batch_chunks,
                    n_epochs=config.n_epochs,
                    progress_callback=_ppo_progress,
                )
            else:
                trainer.train_feedforward_with_callback(
                    buffer,
                    batch_size=config.batch_size,
                    n_epochs=config.n_epochs,
                    progress_callback=_ppo_progress,
                )

            buffer.reset_rollout()

            sps = (global_step - initial_global_step) / max(time.time() - start_time, 1e-6)
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/SPS", sps, global_step)

            mean_train_return = np.mean(recent_returns) if recent_returns else float("nan")
            mean_train_length = np.mean(recent_lengths) if recent_lengths else float("nan")
            train_success = np.mean(recent_successes) if recent_successes else float("nan")
            pbar.set_postfix_str(
                _progress_status(
                    phase="ready",
                    global_step=global_step,
                    initial_step=initial_global_step,
                    start_time=start_time,
                    optimizer=optimizer,
                    recent_returns=recent_returns,
                    recent_successes=recent_successes,
                )
            )

            if (update + 1) % config.log_interval == 0 or update == 0:
                tqdm.write(
                    f"Update {update + 1:>5d}/{num_updates:<5d} | "
                    f"step {global_step:>9d}/{config.total_steps:<9d} | "
                    f"SPS {sps:>7.1f} | "
                    f"lr {optimizer.param_groups[0]['lr']:.2e} | "
                    f"train_return {mean_train_return:>7.3f} | "
                    f"train_success {train_success:>6.2%} | "
                    f"train_len {mean_train_length:>6.1f}"
                )

            if global_step >= next_eval:
                def _eval_progress(done_episodes: int, total_episodes: int) -> None:
                    pbar.set_postfix_str(
                        _progress_status(
                            phase=f"eval {done_episodes}/{total_episodes}",
                            global_step=global_step,
                            initial_step=initial_global_step,
                            start_time=start_time,
                            optimizer=optimizer,
                            recent_returns=recent_returns,
                            recent_successes=recent_successes,
                        )
                    )
                pbar.set_postfix_str(
                    _progress_status(
                        phase=f"eval 0/{config.eval_episodes}",
                        global_step=global_step,
                        initial_step=initial_global_step,
                        start_time=start_time,
                        optimizer=optimizer,
                        recent_returns=recent_returns,
                        recent_successes=recent_successes,
                    )
                )
                success_rate, mean_return, mean_length = evaluate(
                    model,
                    device,
                    config,
                    progress_callback=_eval_progress,
                )
                writer.add_scalar("eval/success_rate", success_rate, global_step)
                writer.add_scalar("eval/mean_return", mean_return, global_step)
                writer.add_scalar("eval/mean_length", mean_length, global_step)
                tqdm.write(
                    f"Eval step {global_step:>9d} | "
                    f"success {success_rate:>6.2%} | "
                    f"return {mean_return:>6.3f} | "
                    f"len {mean_length:>5.1f} | "
                    f"SPS {sps}"
                )
                next_eval += config.eval_interval

            if global_step >= next_save:
                pbar.set_postfix_str(
                    _progress_status(
                        phase="save",
                        global_step=global_step,
                        initial_step=initial_global_step,
                        start_time=start_time,
                        optimizer=optimizer,
                        recent_returns=recent_returns,
                        recent_successes=recent_successes,
                    )
                )
                ckpt_path = os.path.join(run_dir, f"model_{global_step}.pt")
                _save_checkpoint(ckpt_path, model, optimizer, config, global_step, action_dim)
                latest_path = os.path.join(run_dir, "model_latest.pt")
                _save_checkpoint(latest_path, model, optimizer, config, global_step, action_dim)
                tqdm.write(f"Saved checkpoint: {ckpt_path}")
                next_save += config.save_interval

    final_path = os.path.join(run_dir, "model_final.pt")
    _save_checkpoint(final_path, model, optimizer, config, global_step, action_dim)
    latest_path = os.path.join(run_dir, "model_latest.pt")
    _save_checkpoint(latest_path, model, optimizer, config, global_step, action_dim)
    print(f"Training complete. Final model saved to {final_path}")

    for env in envs:
        env.close()
    writer.close()


def evaluate(
    model,
    device: torch.device,
    config: Config,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[float, float, float]:
    """Evaluate with greedy actions."""

    model.eval()
    successes = 0
    total_return = 0.0
    total_length = 0

    for episode in range(config.eval_episodes):
        env = make_env(config.env_id, seed=config.seed + 10_000 + episode)
        obs_dict, _ = env.reset(seed=config.seed + 10_000 + episode)
        obs = obs_dict["obs"]
        direction = obs_dict["direction"]
        prev_action = obs_dict["prev_action"]
        prev_reward = obs_dict["prev_reward"]
        episode_start = obs_dict["episode_start"]

        obs_ctx = deque(maxlen=config.context_len)
        dir_ctx = deque(maxlen=config.context_len)
        act_ctx = deque(maxlen=config.context_len)
        rew_ctx = deque(maxlen=config.context_len)
        start_ctx = deque(maxlen=config.context_len)

        done = False
        ep_return = 0.0
        ep_len = 0
        last_reward = 0.0

        while not done:
            if config.model in {"mamba", "lstm", "attention"}:
                obs_ctx.append(obs)
                dir_ctx.append(direction)
                act_ctx.append(prev_action)
                rew_ctx.append(prev_reward)
                start_ctx.append(episode_start)
                action = _select_sequence_action(
                    model,
                    device,
                    obs_ctx,
                    dir_ctx,
                    act_ctx,
                    rew_ctx,
                    start_ctx,
                    config.context_len,
                    greedy=True,
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
                    action = torch.argmax(logits, dim=-1).item()

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

        total_return += ep_return
        total_length += ep_len
        successes += int(last_reward > 0)
        env.close()
        if progress_callback is not None:
            progress_callback(episode + 1, config.eval_episodes)

    model.train()
    return (
        successes / config.eval_episodes,
        total_return / config.eval_episodes,
        total_length / config.eval_episodes,
    )


def _bootstrap_value(
    model,
    config: Config,
    buffer: RolloutBuffer,
    device: torch.device,
    obs_buf: np.ndarray,
    direction_buf: np.ndarray,
    prev_action_buf: np.ndarray,
    prev_reward_buf: np.ndarray,
    episode_start_buf: np.ndarray,
) -> np.ndarray:
    with torch.no_grad():
        if config.model in {"mamba", "lstm", "attention"}:
            context = buffer.get_context(obs_buf, direction_buf, prev_action_buf, prev_reward_buf, episode_start_buf)
            _, values = model.forward(
                torch.as_tensor(context[0], device=device),
                torch.as_tensor(context[1], device=device),
                torch.as_tensor(context[2], device=device),
                torch.as_tensor(context[3], device=device),
                torch.as_tensor(context[4], device=device),
            )
            return values[:, -1].cpu().numpy()

        _, next_value = model.forward(
            torch.as_tensor(obs_buf, device=device),
            torch.as_tensor(direction_buf, device=device),
            torch.as_tensor(prev_action_buf, device=device),
            torch.as_tensor(prev_reward_buf, device=device),
            torch.as_tensor(episode_start_buf, device=device),
        )
        return next_value.cpu().numpy()


def _select_sequence_action(
    model,
    device: torch.device,
    obs_ctx: deque,
    dir_ctx: deque,
    act_ctx: deque,
    rew_ctx: deque,
    start_ctx: deque,
    context_len: int,
    *,
    greedy: bool,
) -> int:
    obs_seq, dir_seq, act_seq, rew_seq, start_seq = _pack_context(
        obs_ctx, dir_ctx, act_ctx, rew_ctx, start_ctx, context_len
    )
    with torch.no_grad():
        logits, _ = model.forward(
            torch.as_tensor(obs_seq, device=device),
            torch.as_tensor(dir_seq, device=device),
            torch.as_tensor(act_seq, device=device),
            torch.as_tensor(rew_seq, device=device),
            torch.as_tensor(start_seq, device=device),
        )
        last_logits = logits[:, -1]
        if greedy:
            return torch.argmax(last_logits, dim=-1).item()
        return Categorical(logits=last_logits).sample().item()


def _pack_context(
    obs_ctx: deque,
    dir_ctx: deque,
    act_ctx: deque,
    rew_ctx: deque,
    start_ctx: deque,
    context_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(obs_ctx)
    obs_seq = np.zeros((1, context_len, 7, 7, 3), dtype=np.float32)
    dir_seq = np.zeros((1, context_len, 1), dtype=np.int64)
    act_seq = np.zeros((1, context_len, 7), dtype=np.float32)
    rew_seq = np.zeros((1, context_len, 1), dtype=np.float32)
    start_seq = np.zeros((1, context_len, 1), dtype=np.float32)

    if n:
        obs_seq[0, -n:] = np.asarray(list(obs_ctx), dtype=np.float32)
        dir_seq[0, -n:] = np.asarray(list(dir_ctx), dtype=np.int64)
        act_seq[0, -n:] = np.asarray(list(act_ctx), dtype=np.float32)
        rew_seq[0, -n:] = np.asarray(list(rew_ctx), dtype=np.float32)
        start_seq[0, -n:] = np.asarray(list(start_ctx), dtype=np.float32)

    return obs_seq, dir_seq, act_seq, rew_seq, start_seq


def _save_checkpoint(path: str, model, optimizer, config: Config, global_step: int, action_dim: int) -> None:
    config_dict = asdict(config)
    config_dict["action_dim"] = action_dim
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config_dict": config_dict,
            "global_step": global_step,
        },
        path,
    )


def _default_run_name(config: Config) -> str:
    env_name = config.env_id.replace("MiniGrid-", "").replace("-v0", "")
    model_name = config.model
    if config.model == "mamba" and config.mamba_variant != "mamba":
        model_name = config.mamba_variant
    return f"{model_name}_{env_name}_seed{config.seed}"


def _format_config(config: Config) -> str:
    return "\n".join(f"{key}: {value}" for key, value in asdict(config).items())


def _next_interval_boundary(current_step: int, interval: int) -> int:
    if interval <= 0:
        return sys.maxsize
    return ((current_step // interval) + 1) * interval


def _progress_status(
    *,
    phase: str,
    global_step: int,
    initial_step: int,
    start_time: float,
    optimizer: torch.optim.Optimizer,
    recent_returns: deque,
    recent_successes: deque,
) -> str:
    elapsed = max(time.time() - start_time, 1e-6)
    sps = (global_step - initial_step) / elapsed
    mean_return = np.mean(recent_returns) if recent_returns else float("nan")
    success = np.mean(recent_successes) if recent_successes else float("nan")
    return (
        f"{phase} | sps={sps:.1f} | lr={optimizer.param_groups[0]['lr']:.2e} | "
        f"return={mean_return:.3f} | success={success:.1%}"
    )


def parse_args(default_model: str = "mamba") -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="MiniGrid-MemoryS11-v0")
    parser.add_argument("--model", type=str, default=default_model, choices=["mamba", "attention", "lstm", "mlp"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--no-anneal-lr", dest="anneal_lr", action="store_false")
    parser.set_defaults(anneal_lr=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--ent-coef-final", type=float, default=0.001)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--context-len", type=int, default=64)
    parser.add_argument("--chunk-len", type=int, default=64)
    parser.add_argument("--batch-chunks", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--spatial-encoder", type=str, default="hybrid", choices=["hybrid", "transformer"])
    parser.add_argument("--spatial-layers", type=int, default=2)
    parser.add_argument("--spatial-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--mamba-variant", type=str, default="mamba", choices=["mamba", "mamba2", "mamba3"])
    parser.add_argument("--mamba-layers", type=int, default=2)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--attention-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument(
        "--valid-actions",
        type=str,
        default="0,1,2",
        help="Comma-separated action ids kept in the policy distribution; MiniGrid Memory needs left,right,forward.",
    )
    parser.add_argument("--eval-interval", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--save-interval", type=int, default=100_000)
    parser.add_argument("--log-interval", type=int, default=10, help="Print training progress every N PPO updates")
    parser.add_argument("--no-progress-bar", dest="progress_bar", action="store_false")
    parser.set_defaults(progress_bar=True)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--resume-from", type=str, default="")
    args = parser.parse_args()

    return Config(
        env_id=args.env_id,
        model=args.model,
        seed=args.seed,
        total_steps=args.total_steps,
        learning_rate=args.lr,
        anneal_lr=args.anneal_lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        ent_coef_final=args.ent_coef_final,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        context_len=args.context_len,
        chunk_len=args.chunk_len,
        batch_chunks=args.batch_chunks,
        d_model=args.d_model,
        spatial_encoder=args.spatial_encoder,
        spatial_layers=args.spatial_layers,
        spatial_heads=args.spatial_heads,
        dropout=args.dropout,
        lstm_layers=args.lstm_layers,
        mamba_variant=args.mamba_variant,
        mamba_layers=args.mamba_layers,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        attention_layers=args.attention_layers,
        attention_heads=args.attention_heads,
        valid_actions=args.valid_actions,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        progress_bar=args.progress_bar,
        run_name=args.run_name,
        resume_from=args.resume_from,
    )


def main(default_model: str = "mamba") -> None:
    train(parse_args(default_model=default_model))


if __name__ == "__main__":
    main()
