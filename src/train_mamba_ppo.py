"""Train PPO agents for MiniGrid Memory tasks.

Examples:
  python src/train_mamba_ppo.py --model mamba --env-id MiniGrid-MemoryS11-v0
  python src/train_mamba_ppo.py --model attention --env-id MiniGrid-MemoryS11-v0
  python src/train_mamba_ppo.py --model lstm --env-id MiniGrid-MemoryS11-v0
  python src/train_mamba_ppo.py --model mlp --env-id MiniGrid-MemoryS11-v0
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.metadata
import json
import os
import subprocess
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

SEQUENCE_MODELS = {"mamba", "lstm", "gru", "attention", "gated_attention"}


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
    clip_vloss: bool = False
    norm_adv: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    ent_coef_final: float = 0.001
    spinning_penalty: float = 0.0
    spinning_threshold: int = 10

    num_envs: int = 16
    num_steps: int = 128
    batch_size: int = 256
    n_epochs: int = 4

    context_len: int = 128
    chunk_len: int = 64
    batch_chunks: int = 8

    d_model: int = 128
    spatial_encoder: str = "hybrid"
    spatial_layers: int = 2
    spatial_heads: int = 4
    dropout: float = 0.0
    lstm_layers: int = 1
    gru_layers: int = 1
    mamba_variant: str = "mamba"
    mamba_layers: int = 2
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    mamba_headdim: int = 64
    mamba_ngroups: int = 1
    mamba_chunk_size: int = 64
    mamba_rope_fraction: float = 0.5
    attention_layers: int = 2
    attention_heads: int = 4
    gated_attention_pos: str = "learned"
    slot_count: int = 4
    slot_extractor: str = "iterative"
    slot_iters: int = 3
    slot_mlp_ratio: float = 2.0
    temporal_token_mode: str = "fuse"
    memory_kind: str = "episodic_cue"
    memory_slots: int = 16
    memory_topk: int = 4
    memory_write_window: int = 12
    aux_recall_coef: float = 0.05
    valid_actions: str = "0,1,2"
    stateful_rollout: bool = True
    torch_compile: bool = False
    amp: str = "none"
    allow_legacy_load: bool = False

    curriculum: bool = False
    curriculum_envs: str = "MiniGrid-MemoryS11-v0,MiniGrid-MemoryS13-v0,MiniGrid-MemoryS13Random-v0,MiniGrid-MemoryS17Random-v0"
    curriculum_thresholds: str = "0.90,0.85,0.80"
    curriculum_patience: int = 3

    eval_interval: int = 20_000
    eval_episodes: int = 30
    save_interval: int = 100_000
    log_interval: int = 10
    progress_bar: bool = True
    run_name: str = ""
    resume_from: str = ""
    transfer_from: str = ""


def train(config: Config) -> None:
    if config.env_id not in MEMORY_ENVS and "Memory" in config.env_id:
        print(f"Warning: {config.env_id} is not in the known Memory env list.")
    if config.model in SEQUENCE_MODELS and config.chunk_len > config.num_steps:
        raise ValueError("--chunk-len must be <= --num-steps for sequence models.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    amp_dtype = _amp_dtype(config.amp, device)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    curriculum_envs = _parse_str_list(config.curriculum_envs) if config.curriculum else [config.env_id]
    curriculum_thresholds = _parse_float_list(config.curriculum_thresholds)
    curriculum_stage = 0
    curriculum_hits = 0
    if config.curriculum:
        config.env_id = curriculum_envs[0]

    run_name = config.run_name or _default_run_name(config)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_dir = os.path.join(project_root, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    writer = SummaryWriter(run_dir)
    writer.add_text("config", _format_config(config), 0)

    global_step = 0
    ckpt = None
    if config.resume_from and config.transfer_from:
        raise ValueError("Use only one of --resume-from or --transfer-from.")
    if config.resume_from:
        ckpt = torch.load(config.resume_from, map_location=device, weights_only=False)
        global_step = int(ckpt.get("global_step", 0))
        print(f"Resumed checkpoint {config.resume_from} at global_step={global_step}")
        _apply_checkpoint_arch_config(config, ckpt)
    elif config.transfer_from:
        ckpt = torch.load(config.transfer_from, map_location=device, weights_only=False)
        print(f"Transferred model weights from {config.transfer_from}; optimizer, scheduler, env, and global_step were not restored.")

    envs = [
        make_env(
            config.env_id,
            seed=config.seed + i,
            spinning_penalty=config.spinning_penalty,
            spinning_threshold=config.spinning_threshold,
        )
        for i in range(config.num_envs)
    ]

    first_obs, _ = envs[0].reset(seed=config.seed)
    obs_shape = first_obs["obs"].shape
    action_dim = envs[0].action_space.n

    model = build_actor_critic(config, action_dim=action_dim).to(device)
    if config.resume_from or config.transfer_from:
        missing, unexpected = model.load_state_dict(
            ckpt["model_state_dict"],
            strict=not config.allow_legacy_load,
        )
        if config.allow_legacy_load and (missing or unexpected):
            print(f"Checkpoint loaded with missing={missing} unexpected={unexpected}")
    _write_run_config(run_dir, config, action_dim)

    if config.torch_compile:
        if hasattr(torch, "compile"):
            model.forward = torch.compile(model.forward)
            print("torch.compile enabled for model.forward")
        else:
            print("Warning: --compile requested, but this PyTorch build has no torch.compile.")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
    optimizer_loaded = False
    if config.resume_from and "optimizer_state_dict" in ckpt:
        optimizer_loaded = _try_load_optimizer_state(optimizer, ckpt["optimizer_state_dict"])

    remaining_steps = max(config.total_steps - global_step, config.num_envs * config.num_steps)
    num_updates = max(1, remaining_steps // (config.num_envs * config.num_steps))
    scheduler = None
    if config.anneal_lr:
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_updates,
            eta_min=1e-6,
        )
        if config.resume_from and "scheduler_state_dict" in ckpt:
            if optimizer_loaded:
                _try_load_scheduler_state(scheduler, ckpt["scheduler_state_dict"])
            else:
                print("Scheduler state skipped because optimizer state was rebuilt.")

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
        clip_vloss=config.clip_vloss,
        norm_adv=config.norm_adv,
        amp_dtype=amp_dtype,
        aux_recall_coef=config.aux_recall_coef,
    )

    obs_buf = np.zeros((config.num_envs, *obs_shape), dtype=np.uint8)
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
    dynamic_sequence_len = _dynamic_sequence_len_enabled(config)
    mamba_stateful_supported = not (config.model == "mamba" and config.mamba_variant == "mamba3")
    use_stateful_rollout = (
        config.stateful_rollout
        and mamba_stateful_supported
        and config.model in {"mamba", "gated_attention"}
        and (config.model != "gated_attention" or config.gated_attention_pos in {"none", "alibi"})
        and hasattr(model, "init_inference_state")
        and hasattr(model, "get_action_and_value_step")
    )
    mamba_inference_state = None
    if use_stateful_rollout:
        try:
            mamba_inference_state = model.init_inference_state(
                config.num_envs,
                device=device,
                dtype=amp_dtype or _model_parameter_dtype(model),
            )
            print(f"Stateful {config.model} rollout enabled.")
        except Exception as exc:
            use_stateful_rollout = False
            print(f"Stateful {config.model} rollout disabled during cache init: {type(exc).__name__}: {exc}")

    print(
        "Training config: "
        f"run={run_name} env={config.env_id} model={config.model}/{config.mamba_variant} "
        f"steps={global_step}->{config.total_steps} num_envs={config.num_envs} "
        f"rollout={config.num_steps} context={config.context_len} "
        f"chunk={config.chunk_len} batch_chunks={config.batch_chunks} "
        f"spatial={config.spatial_encoder} valid_actions={config.valid_actions} "
        f"slots={config.slot_extractor}/{config.slot_count} token_mode={config.temporal_token_mode} "
        f"memory={config.memory_kind} aux={config.aux_recall_coef} "
        f"gated_pos={config.gated_attention_pos} "
        f"amp={config.amp} compile={config.torch_compile} stateful={use_stateful_rollout} "
        f"dynamic_seq={dynamic_sequence_len}",
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
            progress = min(global_step / max(config.total_steps, 1), 1.0)
            trainer.ent_coef = config.ent_coef + progress * (config.ent_coef_final - config.ent_coef)

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
                if use_stateful_rollout and mamba_inference_state is not None:
                    try:
                        with torch.no_grad(), _autocast_context(device, amp_dtype):
                            action, logprob, _, value, mamba_inference_state = model.get_action_and_value_step(
                                torch.as_tensor(obs_buf, device=device),
                                torch.as_tensor(direction_buf, device=device),
                                torch.as_tensor(prev_action_buf, device=device),
                                torch.as_tensor(prev_reward_buf, device=device),
                                torch.as_tensor(episode_start_buf, device=device),
                                mamba_inference_state,
                            )
                    except (AssertionError, FloatingPointError, NotImplementedError, RuntimeError) as exc:
                        use_stateful_rollout = False
                        mamba_inference_state = None
                        tqdm.write(
                            f"Stateful {config.model} rollout disabled; falling back to full-context rollout "
                            f"after {type(exc).__name__}: {exc}"
                        )
                        context = buffer.get_context(
                            obs_buf,
                            direction_buf,
                            prev_action_buf,
                            prev_reward_buf,
                            episode_start_buf,
                        )
                        with torch.no_grad(), _autocast_context(device, amp_dtype):
                            action, logprob, _, value = model.get_action_and_value(
                                torch.as_tensor(context[0], device=device),
                                torch.as_tensor(context[1], device=device),
                                torch.as_tensor(context[2], device=device),
                                torch.as_tensor(context[3], device=device),
                                torch.as_tensor(context[4], device=device),
                                valid_mask=torch.as_tensor(context[5], device=device),
                            )
                elif config.model in SEQUENCE_MODELS:
                    context = buffer.get_context(
                        obs_buf,
                        direction_buf,
                        prev_action_buf,
                        prev_reward_buf,
                        episode_start_buf,
                    )
                    with torch.no_grad(), _autocast_context(device, amp_dtype):
                        action, logprob, _, value = model.get_action_and_value(
                            torch.as_tensor(context[0], device=device),
                            torch.as_tensor(context[1], device=device),
                            torch.as_tensor(context[2], device=device),
                            torch.as_tensor(context[3], device=device),
                            torch.as_tensor(context[4], device=device),
                            valid_mask=torch.as_tensor(context[5], device=device),
                        )
                else:
                    with torch.no_grad(), _autocast_context(device, amp_dtype):
                        action, logprob, _, value = model.get_action_and_value(
                            torch.as_tensor(obs_buf, device=device),
                            torch.as_tensor(direction_buf, device=device),
                            torch.as_tensor(prev_action_buf, device=device),
                            torch.as_tensor(prev_reward_buf, device=device),
                            torch.as_tensor(episode_start_buf, device=device),
                        )

                action_np = action.cpu().numpy().astype(np.int64, copy=False)
                logprob_np = logprob.float().cpu().numpy()
                value_np = value.float().cpu().numpy()
                rollout_obs = obs_buf.copy()
                rollout_directions = direction_buf.copy()
                rollout_prev_actions = prev_action_buf.copy()
                rollout_prev_rewards = prev_reward_buf.copy()
                rollout_episode_starts = episode_start_buf.copy()
                rewards_np = np.zeros(config.num_envs, dtype=np.float32)
                done_mask_np = np.zeros(config.num_envs, dtype=bool)

                for env_idx, env in enumerate(envs):
                    obs_dict, reward, terminated, truncated, _ = env.step(int(action_np[env_idx]))
                    done = terminated or truncated
                    rewards_np[env_idx] = float(reward)
                    done_mask_np[env_idx] = done

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

                buffer.add_batch(
                    obs=rollout_obs,
                    directions=rollout_directions,
                    actions=action_np,
                    logprobs=logprob_np,
                    values=value_np,
                    rewards=rewards_np,
                    dones=done_mask_np,
                    prev_actions=rollout_prev_actions,
                    prev_rewards=rollout_prev_rewards,
                    episode_starts=rollout_episode_starts,
                )

                if use_stateful_rollout and mamba_inference_state is not None and done_mask_np.any():
                    model.reset_inference_state(
                        mamba_inference_state,
                        torch.as_tensor(done_mask_np, device=device, dtype=torch.bool),
                    )

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
                amp_dtype,
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
            if update == 0 or (update + 1) % config.log_interval == 0:
                tqdm.write(
                    f"PPO update start {update + 1}/{num_updates} | "
                    f"step {global_step}/{config.total_steps} | "
                    f"chunk={config.chunk_len} batch_chunks={config.batch_chunks} "
                    f"epochs={config.n_epochs} dynamic_seq={dynamic_sequence_len}"
                )
            if config.model in SEQUENCE_MODELS:
                trainer.train_sequence_with_callback(
                    buffer,
                    chunk_len=config.chunk_len,
                    batch_chunks=config.batch_chunks,
                    n_epochs=config.n_epochs,
                    dynamic_sequence_len=dynamic_sequence_len,
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
            if scheduler is not None:
                scheduler.step()

            sps = (global_step - initial_global_step) / max(time.time() - start_time, 1e-6)
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/SPS", sps, global_step)
            for name, value in _memory_diagnostics(model).items():
                writer.add_scalar(f"memory/{name}", value, global_step)

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
                    amp_dtype=amp_dtype,
                    progress_callback=_eval_progress,
                )
                writer.add_scalar("eval/success_rate", success_rate, global_step)
                writer.add_scalar("eval/mean_return", mean_return, global_step)
                writer.add_scalar("eval/mean_length", mean_length, global_step)
                writer.add_scalar("curriculum/stage", curriculum_stage, global_step)
                tqdm.write(
                    f"Eval step {global_step:>9d} | "
                    f"success {success_rate:>6.2%} | "
                    f"return {mean_return:>6.3f} | "
                    f"len {mean_length:>5.1f} | "
                    f"SPS {sps}"
                )
                if config.curriculum and curriculum_stage < len(curriculum_envs) - 1:
                    threshold = curriculum_thresholds[min(curriculum_stage, len(curriculum_thresholds) - 1)]
                    curriculum_hits = curriculum_hits + 1 if success_rate >= threshold else 0
                    writer.add_scalar("curriculum/hits", curriculum_hits, global_step)
                    if curriculum_hits >= config.curriculum_patience:
                        curriculum_stage += 1
                        curriculum_hits = 0
                        config.env_id = curriculum_envs[curriculum_stage]
                        tqdm.write(f"Curriculum advanced to stage {curriculum_stage}: {config.env_id}")
                        for env in envs:
                            env.close()
                        envs = [
                            make_env(
                                config.env_id,
                                seed=config.seed + 10_000 * (curriculum_stage + 1) + i,
                                spinning_penalty=config.spinning_penalty,
                                spinning_threshold=config.spinning_threshold,
                            )
                            for i in range(config.num_envs)
                        ]
                        _reset_env_buffers(
                            envs,
                            config,
                            obs_buf,
                            direction_buf,
                            prev_action_buf,
                            prev_reward_buf,
                            episode_start_buf,
                            seed_offset=10_000 * (curriculum_stage + 1),
                        )
                        buffer = RolloutBuffer(
                            num_envs=config.num_envs,
                            num_steps=config.num_steps,
                            obs_shape=obs_shape,
                            action_dim=action_dim,
                            context_len=config.context_len,
                        )
                        episode_returns.fill(0.0)
                        episode_lengths.fill(0)
                        recent_returns.clear()
                        recent_lengths.clear()
                        recent_successes.clear()
                        if use_stateful_rollout:
                            mamba_inference_state = model.init_inference_state(
                                config.num_envs,
                                device=device,
                                dtype=amp_dtype or _model_parameter_dtype(model),
                            )
                        _write_run_config(run_dir, config, action_dim)
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
                _save_checkpoint(ckpt_path, model, optimizer, scheduler, config, global_step, action_dim)
                latest_path = os.path.join(run_dir, "model_latest.pt")
                _save_checkpoint(latest_path, model, optimizer, scheduler, config, global_step, action_dim)
                tqdm.write(f"Saved checkpoint: {ckpt_path}")
                next_save += config.save_interval

    final_path = os.path.join(run_dir, "model_final.pt")
    _save_checkpoint(final_path, model, optimizer, scheduler, config, global_step, action_dim)
    latest_path = os.path.join(run_dir, "model_latest.pt")
    _save_checkpoint(latest_path, model, optimizer, scheduler, config, global_step, action_dim)
    print(f"Training complete. Final model saved to {final_path}")

    for env in envs:
        env.close()
    writer.close()


def evaluate(
    model,
    device: torch.device,
    config: Config,
    amp_dtype: torch.dtype | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[float, float, float]:
    """Evaluate with greedy actions."""

    model.eval()
    successes = 0
    total_return = 0.0
    total_length = 0
    use_stateful_eval = _stateful_eval_enabled(model, config)
    if use_stateful_eval:
        try:
            return _evaluate_stateful_batched(model, device, config, amp_dtype, progress_callback)
        except Exception as exc:
            print(f"Warning: stateful batched eval failed; falling back to full-context eval. Reason: {type(exc).__name__}: {exc}")

    for episode in range(config.eval_episodes):
        env = make_env(
            config.env_id,
            seed=config.seed + 10_000 + episode,
            spinning_penalty=config.spinning_penalty,
            spinning_threshold=config.spinning_threshold,
        )
        obs_dict, _ = env.reset(seed=config.seed + 10_000 + episode)
        action_dim = env.action_space.n
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
            if config.model in SEQUENCE_MODELS:
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
                    action_dim,
                    greedy=True,
                    amp_dtype=amp_dtype,
                )
            else:
                with torch.no_grad(), _autocast_context(device, amp_dtype):
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


def _evaluate_stateful_batched(
    model,
    device: torch.device,
    config: Config,
    amp_dtype: torch.dtype | None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[float, float, float]:
    envs = [
        make_env(
            config.env_id,
            seed=config.seed + 10_000 + episode,
            spinning_penalty=config.spinning_penalty,
            spinning_threshold=config.spinning_threshold,
        )
        for episode in range(config.eval_episodes)
    ]

    first_obs, _ = envs[0].reset(seed=config.seed + 10_000)
    obs_shape = first_obs["obs"].shape
    action_dim = envs[0].action_space.n
    obs_buf = np.zeros((config.eval_episodes, *obs_shape), dtype=np.uint8)
    direction_buf = np.zeros((config.eval_episodes, 1), dtype=np.int64)
    prev_action_buf = np.zeros((config.eval_episodes, action_dim), dtype=np.float32)
    prev_reward_buf = np.zeros((config.eval_episodes, 1), dtype=np.float32)
    episode_start_buf = np.ones((config.eval_episodes, 1), dtype=np.float32)

    obs_buf[0] = first_obs["obs"]
    direction_buf[0] = first_obs["direction"]
    prev_action_buf[0] = first_obs["prev_action"]
    prev_reward_buf[0] = first_obs["prev_reward"]
    episode_start_buf[0] = first_obs["episode_start"]
    for episode in range(1, config.eval_episodes):
        obs_dict, _ = envs[episode].reset(seed=config.seed + 10_000 + episode)
        obs_buf[episode] = obs_dict["obs"]
        direction_buf[episode] = obs_dict["direction"]
        prev_action_buf[episode] = obs_dict["prev_action"]
        prev_reward_buf[episode] = obs_dict["prev_reward"]
        episode_start_buf[episode] = obs_dict["episode_start"]

    inference_state = model.init_inference_state(
        config.eval_episodes,
        device=device,
        dtype=amp_dtype or _model_parameter_dtype(model),
    )
    done_mask = np.zeros(config.eval_episodes, dtype=bool)
    episode_returns = np.zeros(config.eval_episodes, dtype=np.float32)
    episode_lengths = np.zeros(config.eval_episodes, dtype=np.int32)
    last_rewards = np.zeros(config.eval_episodes, dtype=np.float32)
    completed = 0

    try:
        while completed < config.eval_episodes:
            with torch.no_grad(), _autocast_context(device, amp_dtype):
                action, _, _, _, inference_state = model.get_action_and_value_step(
                    torch.as_tensor(obs_buf, device=device),
                    torch.as_tensor(direction_buf, device=device),
                    torch.as_tensor(prev_action_buf, device=device),
                    torch.as_tensor(prev_reward_buf, device=device),
                    torch.as_tensor(episode_start_buf, device=device),
                    inference_state,
                    deterministic=True,
                )
            action_np = action.cpu().numpy().astype(np.int64, copy=False)

            for episode, env in enumerate(envs):
                if done_mask[episode]:
                    continue
                obs_dict, reward, terminated, truncated, _ = env.step(int(action_np[episode]))
                done = terminated or truncated
                episode_returns[episode] += float(reward)
                episode_lengths[episode] += 1
                last_rewards[episode] = float(reward)
                if done:
                    done_mask[episode] = True
                    completed += 1
                    if progress_callback is not None:
                        progress_callback(completed, config.eval_episodes)
                    continue

                obs_buf[episode] = obs_dict["obs"]
                direction_buf[episode] = obs_dict["direction"]
                prev_action_buf[episode] = obs_dict["prev_action"]
                prev_reward_buf[episode] = obs_dict["prev_reward"]
                episode_start_buf[episode] = obs_dict["episode_start"]
    finally:
        for env in envs:
            env.close()

    return (
        float(np.mean(last_rewards > 0.0)),
        float(np.mean(episode_returns)),
        float(np.mean(episode_lengths)),
    )


def _stateful_eval_enabled(model, config: Config) -> bool:
    if not config.stateful_rollout:
        return False
    if config.model == "mamba" and config.mamba_variant == "mamba3":
        return False
    if config.model not in {"mamba", "gated_attention"}:
        return False
    if config.model == "gated_attention" and config.gated_attention_pos not in {"none", "alibi"}:
        return False
    return hasattr(model, "init_inference_state") and hasattr(model, "get_action_and_value_step")


def _reset_env_buffers(
    envs,
    config: Config,
    obs_buf: np.ndarray,
    direction_buf: np.ndarray,
    prev_action_buf: np.ndarray,
    prev_reward_buf: np.ndarray,
    episode_start_buf: np.ndarray,
    *,
    seed_offset: int = 0,
) -> None:
    for env_idx, env in enumerate(envs):
        obs_dict, _ = env.reset(seed=config.seed + seed_offset + env_idx)
        obs_buf[env_idx] = obs_dict["obs"]
        direction_buf[env_idx] = obs_dict["direction"]
        prev_action_buf[env_idx] = obs_dict["prev_action"]
        prev_reward_buf[env_idx] = obs_dict["prev_reward"]
        episode_start_buf[env_idx] = obs_dict["episode_start"]


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
    amp_dtype: torch.dtype | None,
) -> np.ndarray:
    with torch.no_grad(), _autocast_context(device, amp_dtype):
        if config.model in SEQUENCE_MODELS:
            context = buffer.get_context(obs_buf, direction_buf, prev_action_buf, prev_reward_buf, episode_start_buf)
            valid_mask_np = context[5]
            valid_mask_t = None if bool(valid_mask_np.all()) else torch.as_tensor(valid_mask_np, device=device)
            _, values = model.forward(
                torch.as_tensor(context[0], device=device),
                torch.as_tensor(context[1], device=device),
                torch.as_tensor(context[2], device=device),
                torch.as_tensor(context[3], device=device),
                torch.as_tensor(context[4], device=device),
                valid_mask=valid_mask_t,
            )
            if valid_mask_t is None:
                return values[:, -1].float().cpu().numpy()
            mask_t = valid_mask_t
            last_idx = mask_t.long().sum(dim=1).clamp(min=1, max=values.shape[1]) - 1
            rows = torch.arange(values.shape[0], device=device)
            return values[rows, last_idx].float().cpu().numpy()

        _, next_value = model.forward(
            torch.as_tensor(obs_buf, device=device),
            torch.as_tensor(direction_buf, device=device),
            torch.as_tensor(prev_action_buf, device=device),
            torch.as_tensor(prev_reward_buf, device=device),
            torch.as_tensor(episode_start_buf, device=device),
        )
        return next_value.float().cpu().numpy()


def _dynamic_sequence_len_enabled(config: Config) -> bool:
    if config.model == "attention":
        return False
    if config.model == "gated_attention" and config.gated_attention_pos == "learned":
        return False
    return config.model in SEQUENCE_MODELS


def _select_sequence_action(
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
    greedy: bool,
    amp_dtype: torch.dtype | None = None,
) -> int:
    obs_seq, dir_seq, act_seq, rew_seq, start_seq, valid_mask = _pack_context(
        obs_ctx, dir_ctx, act_ctx, rew_ctx, start_ctx, context_len, action_dim
    )
    with torch.no_grad(), _autocast_context(device, amp_dtype):
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


def _save_checkpoint(
    path: str,
    model,
    optimizer,
    scheduler,
    config: Config,
    global_step: int,
    action_dim: int,
) -> None:
    config_dict = asdict(config)
    config_dict["action_dim"] = int(action_dim)
    state = {
        "checkpoint_schema_version": 2,
        "model_state_dict": _model_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "config_dict": config_dict,
        "runtime_metadata": _runtime_metadata(config, action_dim),
        "global_step": global_step,
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(state, path)


def _write_run_config(run_dir: str, config: Config, action_dim: int) -> None:
    path = os.path.join(run_dir, "run_config.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_runtime_metadata(config, action_dim), fh, indent=2, sort_keys=True, default=_json_default)


def _runtime_metadata(config: Config, action_dim: int) -> dict:
    return {
        "checkpoint_schema_version": 2,
        "config": asdict(config),
        "action_dim": int(action_dim),
        "git_commit": _git_commit(),
        "packages": _package_versions(
            "torch",
            "gymnasium",
            "minigrid",
            "mamba-ssm",
            "causal-conv1d",
            "triton",
            "triton-windows",
        ),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "actual_mamba_variant": config.mamba_variant if config.model == "mamba" else None,
    }


def _git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return proc.stdout.strip() or None


def _package_versions(*names: str) -> dict[str, str | None]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.dtype):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _default_run_name(config: Config) -> str:
    env_name = config.env_id.replace("MiniGrid-", "").replace("-v0", "")
    model_name = config.model
    if config.model == "mamba" and config.mamba_variant != "mamba":
        model_name = config.mamba_variant
    if config.memory_kind == "episodic_cue" and config.slot_extractor == "iterative":
        model_name = f"slot_memory_{model_name}"
    return f"{model_name}_{env_name}_seed{config.seed}"


def _parse_str_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_float_list(value: str) -> list[float]:
    values = [float(part) for part in value.split(",") if part.strip()]
    return values or [1.0]


def _format_config(config: Config) -> str:
    return "\n".join(f"{key}: {value}" for key, value in asdict(config).items())


def _next_interval_boundary(current_step: int, interval: int) -> int:
    if interval <= 0:
        return sys.maxsize
    return ((current_step // interval) + 1) * interval


def _amp_dtype(value: str, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda" or value == "none":
        return None
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    raise ValueError(f"Unknown AMP mode: {value}")


def _autocast_context(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _model_parameter_dtype(model) -> torch.dtype:
    return next(model.parameters()).dtype


def _model_state_dict(model):
    return getattr(model, "_orig_mod", model).state_dict()


def _memory_diagnostics(model) -> dict[str, float]:
    base = getattr(model, "_orig_mod", model)
    values: dict[str, list[float]] = {
        "gate_mean": [],
        "retrieval_entropy": [],
        "write_rate": [],
    }
    for module in base.modules():
        if hasattr(module, "last_gate_mean"):
            values["gate_mean"].append(float(module.last_gate_mean))
        if hasattr(module, "last_retrieval_entropy"):
            values["retrieval_entropy"].append(float(module.last_retrieval_entropy))
        if hasattr(module, "last_write_rate"):
            values["write_rate"].append(float(module.last_write_rate))
    return {
        key: float(np.mean(items))
        for key, items in values.items()
        if items
    }


def _apply_checkpoint_arch_config(config: Config, ckpt) -> None:
    saved = ckpt.get("config_dict", {})
    if not isinstance(saved, dict):
        return

    arch_keys = (
        "env_id",
        "model",
        "spatial_encoder",
        "spatial_layers",
        "spatial_heads",
        "slot_count",
        "slot_extractor",
        "slot_iters",
        "slot_mlp_ratio",
        "temporal_token_mode",
        "memory_kind",
        "memory_slots",
        "memory_topk",
        "memory_write_window",
        "aux_recall_coef",
        "mamba_variant",
        "mamba_layers",
        "d_model",
        "d_state",
        "d_conv",
        "expand",
        "mamba_headdim",
        "mamba_ngroups",
        "mamba_chunk_size",
        "mamba_rope_fraction",
        "attention_layers",
        "attention_heads",
        "gru_layers",
        "gated_attention_pos",
        "valid_actions",
    )
    changed = []
    for key in arch_keys:
        if key not in saved or not hasattr(config, key):
            continue
        old_value = getattr(config, key)
        new_value = saved[key]
        if old_value != new_value:
            setattr(config, key, new_value)
            changed.append(f"{key}={new_value}")
    if changed:
        print("Resume architecture config restored from checkpoint: " + ", ".join(changed))


def _try_load_optimizer_state(optimizer, state_dict) -> bool:
    try:
        optimizer.load_state_dict(state_dict)
        print("Optimizer state loaded.")
        return True
    except (KeyError, RuntimeError, ValueError) as exc:
        restored_lr = _restore_optimizer_lr(optimizer, state_dict)
        if restored_lr is None:
            lr_note = "using config learning rate"
        else:
            lr_note = f"preserved checkpoint lr={restored_lr:.3e}"
        print(
            "Warning: optimizer state is incompatible with the current model; "
            f"continuing with fresh optimizer moments ({lr_note}). "
            f"Reason: {type(exc).__name__}: {exc}"
        )
        return False


def _restore_optimizer_lr(optimizer, state_dict) -> float | None:
    param_groups = state_dict.get("param_groups", []) if isinstance(state_dict, dict) else []
    restored_lr = None
    for group, saved_group in zip(optimizer.param_groups, param_groups):
        if "lr" in saved_group:
            group["lr"] = saved_group["lr"]
            restored_lr = float(saved_group["lr"])
    return restored_lr


def _try_load_scheduler_state(scheduler, state_dict) -> bool:
    try:
        scheduler.load_state_dict(state_dict)
        print("Scheduler state loaded.")
        return True
    except (KeyError, RuntimeError, ValueError) as exc:
        print(
            "Warning: scheduler state is incompatible with the current run; "
            f"using a fresh scheduler. Reason: {type(exc).__name__}: {exc}"
        )
        return False


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
    parser.add_argument("--model", type=str, default=default_model, choices=["mamba", "attention", "lstm", "gru", "mlp", "gated_attention"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--no-anneal-lr", dest="anneal_lr", action="store_false")
    parser.set_defaults(anneal_lr=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--clip-vloss", dest="clip_vloss", action="store_true", help="Enable PPO2-style value loss clipping.")
    parser.add_argument("--no-clip-vloss", dest="clip_vloss", action="store_false", help="Keep value loss unclipped; this is the default.")
    parser.set_defaults(clip_vloss=False)
    parser.add_argument("--no-norm-adv", dest="norm_adv", action="store_false")
    parser.set_defaults(norm_adv=True)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--ent-coef-final", type=float, default=0.001)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--spinning-penalty", type=float, default=0.0, help="Penalty per step for staying in the same cell")
    parser.add_argument("--spinning-threshold", type=int, default=10, help="Max steps allowed in the same cell before penalty")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--chunk-len", type=int, default=64)
    parser.add_argument("--batch-chunks", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--spatial-encoder", type=str, default="hybrid", choices=["hybrid", "transformer"])
    parser.add_argument("--spatial-layers", type=int, default=2)
    parser.add_argument("--spatial-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--gru-layers", type=int, default=1)
    parser.add_argument("--mamba-variant", type=str, default="mamba", choices=["mamba", "mamba2", "mamba3"])
    parser.add_argument("--mamba-layers", type=int, default=2)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--mamba-headdim", type=int, default=64)
    parser.add_argument("--mamba-ngroups", type=int, default=1)
    parser.add_argument("--mamba-chunk-size", type=int, default=64)
    parser.add_argument("--mamba-rope-fraction", type=float, default=0.5)
    parser.add_argument("--attention-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--gated-attention-pos", type=str, default="learned", choices=["learned", "none", "alibi"])
    parser.add_argument("--slot-count", type=int, default=4, help="Number of learned spatial slot tokens per step before the global decision token.")
    parser.add_argument("--slot-extractor", type=str, default="iterative", choices=["query_pool", "iterative"])
    parser.add_argument("--slot-iters", type=int, default=3)
    parser.add_argument("--slot-mlp-ratio", type=float, default=2.0)
    parser.add_argument("--temporal-token-mode", type=str, default="fuse", choices=["flatten", "fuse"])
    parser.add_argument("--memory-kind", type=str, default="episodic_cue", choices=["none", "episodic_cue"])
    parser.add_argument("--memory-slots", type=int, default=16)
    parser.add_argument("--memory-topk", type=int, default=4)
    parser.add_argument("--memory-write-window", type=int, default=12)
    parser.add_argument("--aux-recall-coef", type=float, default=0.05)
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
    parser.add_argument("--no-stateful-rollout", dest="stateful_rollout", action="store_false")
    parser.set_defaults(stateful_rollout=True)
    parser.add_argument("--compile", dest="torch_compile", action="store_true", help="Compile model.forward with torch.compile")
    parser.set_defaults(torch_compile=False)
    parser.add_argument("--amp", type=str, default="none", choices=["none", "bf16", "fp16"])
    parser.add_argument("--allow-legacy-load", action="store_true", help="Load checkpoints with strict=False for old, schema-less experiments.")
    parser.set_defaults(allow_legacy_load=False)
    parser.add_argument("--curriculum", action="store_true", help="Train through the Memory curriculum S11 -> S13 -> S13Random -> S17Random.")
    parser.set_defaults(curriculum=False)
    parser.add_argument("--curriculum-envs", type=str, default="MiniGrid-MemoryS11-v0,MiniGrid-MemoryS13-v0,MiniGrid-MemoryS13Random-v0,MiniGrid-MemoryS17Random-v0")
    parser.add_argument("--curriculum-thresholds", type=str, default="0.90,0.85,0.80")
    parser.add_argument("--curriculum-patience", type=int, default=3)
    parser.add_argument("--no-progress-bar", dest="progress_bar", action="store_false")
    parser.set_defaults(progress_bar=True)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--resume-from", type=str, default="")
    parser.add_argument(
        "--transfer-from",
        type=str,
        default="",
        help="Initialize model weights from a checkpoint without restoring env, global step, optimizer, or scheduler.",
    )
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
        clip_vloss=args.clip_vloss,
        norm_adv=args.norm_adv,
        ent_coef=args.ent_coef,
        ent_coef_final=args.ent_coef_final,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
        spinning_penalty=args.spinning_penalty,
        spinning_threshold=args.spinning_threshold,
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
        gru_layers=args.gru_layers,
        mamba_variant=args.mamba_variant,
        mamba_layers=args.mamba_layers,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        mamba_headdim=args.mamba_headdim,
        mamba_ngroups=args.mamba_ngroups,
        mamba_chunk_size=args.mamba_chunk_size,
        mamba_rope_fraction=args.mamba_rope_fraction,
        attention_layers=args.attention_layers,
        attention_heads=args.attention_heads,
        gated_attention_pos=args.gated_attention_pos,
        slot_count=args.slot_count,
        slot_extractor=args.slot_extractor,
        slot_iters=args.slot_iters,
        slot_mlp_ratio=args.slot_mlp_ratio,
        temporal_token_mode=args.temporal_token_mode,
        memory_kind=args.memory_kind,
        memory_slots=args.memory_slots,
        memory_topk=args.memory_topk,
        memory_write_window=args.memory_write_window,
        aux_recall_coef=args.aux_recall_coef,
        valid_actions=args.valid_actions,
        stateful_rollout=args.stateful_rollout,
        torch_compile=args.torch_compile,
        amp=args.amp,
        allow_legacy_load=args.allow_legacy_load,
        curriculum=args.curriculum,
        curriculum_envs=args.curriculum_envs,
        curriculum_thresholds=args.curriculum_thresholds,
        curriculum_patience=args.curriculum_patience,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        progress_bar=args.progress_bar,
        run_name=args.run_name,
        resume_from=args.resume_from,
        transfer_from=args.transfer_from,
    )


def main(default_model: str = "mamba") -> None:
    train(parse_args(default_model=default_model))


if __name__ == "__main__":
    main()
