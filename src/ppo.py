"""Rollout storage and PPO updates."""

from __future__ import annotations

from collections import deque
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from src.models import _safe_categorical


class RolloutBuffer:
    """Fixed-length rollout buffer with per-env context history."""

    def __init__(
        self,
        num_envs: int,
        num_steps: int,
        obs_shape,
        action_dim: int,
        context_len: int = 64,
    ):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.context_len = context_len
        self.action_dim = action_dim

        self.observations = np.zeros((num_steps, num_envs, *obs_shape), dtype=np.float32)
        self.directions = np.zeros((num_steps, num_envs, 1), dtype=np.int64)
        self.actions = np.zeros((num_steps, num_envs), dtype=np.int64)
        self.prev_actions = np.zeros((num_steps, num_envs, action_dim), dtype=np.float32)
        self.prev_rewards = np.zeros((num_steps, num_envs, 1), dtype=np.float32)
        self.episode_starts = np.zeros((num_steps, num_envs, 1), dtype=np.float32)
        self.logprobs = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_envs), dtype=bool)

        self.advantages: np.ndarray | None = None
        self.returns: np.ndarray | None = None

        self.obs_history = [deque(maxlen=context_len) for _ in range(num_envs)]
        self.dir_history = [deque(maxlen=context_len) for _ in range(num_envs)]
        self.act_history = [deque(maxlen=context_len) for _ in range(num_envs)]
        self.rew_history = [deque(maxlen=context_len) for _ in range(num_envs)]
        self.start_history = [deque(maxlen=context_len) for _ in range(num_envs)]

        self.step = 0

    def reset_context(self, env_idx: int) -> None:
        self.obs_history[env_idx].clear()
        self.dir_history[env_idx].clear()
        self.act_history[env_idx].clear()
        self.rew_history[env_idx].clear()
        self.start_history[env_idx].clear()

    def reset_rollout(self) -> None:
        self.step = 0
        self.advantages = None
        self.returns = None

    def add(
        self,
        env_idx: int,
        obs,
        direction,
        action: int,
        logprob: float,
        value: float,
        reward: float,
        done: bool,
        prev_action,
        prev_reward,
        episode_start,
    ) -> None:
        idx = self.step
        self.observations[idx, env_idx] = obs
        self.directions[idx, env_idx] = direction
        self.actions[idx, env_idx] = action
        self.prev_actions[idx, env_idx] = prev_action
        self.prev_rewards[idx, env_idx] = prev_reward
        self.episode_starts[idx, env_idx] = episode_start
        self.logprobs[idx, env_idx] = logprob
        self.rewards[idx, env_idx] = reward
        self.values[idx, env_idx] = value
        self.dones[idx, env_idx] = done

        self.obs_history[env_idx].append(obs)
        self.dir_history[env_idx].append(direction)
        self.act_history[env_idx].append(prev_action)
        self.rew_history[env_idx].append(prev_reward)
        self.start_history[env_idx].append(episode_start)

        if done:
            self.reset_context(env_idx)

    def get_context(
        self,
        current_obs: np.ndarray | None = None,
        current_directions: np.ndarray | None = None,
        current_prev_actions: np.ndarray | None = None,
        current_prev_rewards: np.ndarray | None = None,
        current_episode_starts: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return right-aligned context windows, optionally including current token."""

        ctx_len = self.context_len
        obs_shape = self.observations.shape[2:]
        obs_ctx = np.zeros((self.num_envs, ctx_len, *obs_shape), dtype=np.float32)
        dir_ctx = np.zeros((self.num_envs, ctx_len, 1), dtype=np.int64)
        act_ctx = np.zeros((self.num_envs, ctx_len, self.action_dim), dtype=np.float32)
        rew_ctx = np.zeros((self.num_envs, ctx_len, 1), dtype=np.float32)
        start_ctx = np.zeros((self.num_envs, ctx_len, 1), dtype=np.float32)

        for env_idx in range(self.num_envs):
            obs_list = list(self.obs_history[env_idx])
            dir_list = list(self.dir_history[env_idx])
            act_list = list(self.act_history[env_idx])
            rew_list = list(self.rew_history[env_idx])
            start_list = list(self.start_history[env_idx])

            if current_obs is not None:
                obs_list.append(current_obs[env_idx])
                dir_list.append(current_directions[env_idx])
                act_list.append(current_prev_actions[env_idx])
                rew_list.append(current_prev_rewards[env_idx])
                start_list.append(current_episode_starts[env_idx])

            n = min(len(obs_list), ctx_len)
            if n:
                obs_ctx[env_idx, -n:] = np.asarray(obs_list[-n:], dtype=np.float32)
                dir_ctx[env_idx, -n:] = np.asarray(dir_list[-n:], dtype=np.int64)
                act_ctx[env_idx, -n:] = np.asarray(act_list[-n:], dtype=np.float32)
                rew_ctx[env_idx, -n:] = np.asarray(rew_list[-n:], dtype=np.float32)
                start_ctx[env_idx, -n:] = np.asarray(start_list[-n:], dtype=np.float32)

        return obs_ctx, dir_ctx, act_ctx, rew_ctx, start_ctx

    def compute_gae(self, next_value, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute and store generalized advantage estimates."""

        next_value = np.asarray(next_value, dtype=np.float32)
        advantages = np.zeros_like(self.rewards, dtype=np.float32)
        lastgaelam = np.zeros(self.num_envs, dtype=np.float32)

        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - self.dones[t].astype(np.float32)
                next_values = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t].astype(np.float32)
                next_values = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            lastgaelam = delta + gamma * gae_lambda * next_non_terminal * lastgaelam
            advantages[t] = lastgaelam

        self.advantages = advantages
        self.returns = advantages + self.values
        return self.advantages, self.returns


class PPOTrainer:
    """PPO updates for feedforward and sequence policies."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        writer: SummaryWriter,
        *,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.writer = writer
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.global_step = 0

    def train_feedforward(self, buffer: RolloutBuffer, batch_size: int, n_epochs: int) -> None:
        self.train_feedforward_with_callback(
            buffer,
            batch_size=batch_size,
            n_epochs=n_epochs,
            progress_callback=None,
        )

    def train_feedforward_with_callback(
        self,
        buffer: RolloutBuffer,
        batch_size: int,
        n_epochs: int,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        if buffer.advantages is None or buffer.returns is None:
            raise RuntimeError("Call buffer.compute_gae(...) before PPO updates.")

        b_obs = buffer.observations.reshape(-1, *buffer.observations.shape[2:])
        b_directions = buffer.directions.reshape(-1, 1)
        b_actions = buffer.actions.reshape(-1)
        b_prev_actions = buffer.prev_actions.reshape(-1, buffer.action_dim)
        b_prev_rewards = buffer.prev_rewards.reshape(-1, 1)
        b_episode_starts = buffer.episode_starts.reshape(-1, 1)
        b_logprobs = buffer.logprobs.reshape(-1)
        b_advantages = _normalize(buffer.advantages.reshape(-1))
        b_returns = buffer.returns.reshape(-1)
        b_values = buffer.values.reshape(-1)

        indices = np.arange(b_obs.shape[0])
        total_batches = n_epochs * max(1, (len(indices) + batch_size - 1) // batch_size)
        batch_counter = 0

        for _ in range(n_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                mb_idx = indices[start : start + batch_size]
                if len(mb_idx) == 0:
                    continue

                _, new_logprob, entropy, new_value = self.model.get_action_and_value(
                    torch.as_tensor(b_obs[mb_idx], device=self.device),
                    torch.as_tensor(b_directions[mb_idx], device=self.device),
                    torch.as_tensor(b_prev_actions[mb_idx], device=self.device),
                    torch.as_tensor(b_prev_rewards[mb_idx], device=self.device),
                    torch.as_tensor(b_episode_starts[mb_idx], device=self.device),
                    action=torch.as_tensor(b_actions[mb_idx], device=self.device).long(),
                )

                stop = self._update_minibatch(
                    new_logprob=new_logprob,
                    entropy=entropy,
                    new_value=new_value,
                    old_logprob=torch.as_tensor(b_logprobs[mb_idx], device=self.device),
                    advantages=torch.as_tensor(b_advantages[mb_idx], device=self.device),
                    returns=torch.as_tensor(b_returns[mb_idx], device=self.device),
                    old_values=torch.as_tensor(b_values[mb_idx], device=self.device),
                )
                batch_counter += 1
                if progress_callback is not None:
                    progress_callback(batch_counter, total_batches)
                if stop:
                    return

    def train_sequence(
        self,
        buffer: RolloutBuffer,
        chunk_len: int,
        batch_chunks: int,
        n_epochs: int,
    ) -> None:
        self.train_sequence_with_callback(
            buffer,
            chunk_len=chunk_len,
            batch_chunks=batch_chunks,
            n_epochs=n_epochs,
            progress_callback=None,
        )

    def train_sequence_with_callback(
        self,
        buffer: RolloutBuffer,
        chunk_len: int,
        batch_chunks: int,
        n_epochs: int,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        if buffer.advantages is None or buffer.returns is None:
            raise RuntimeError("Call buffer.compute_gae(...) before PPO updates.")

        chunk_len = min(chunk_len, buffer.num_steps)
        chunks = _episode_bounded_chunks(buffer, chunk_len)
        if not chunks:
            chunks = [(env_idx, 0, buffer.num_steps) for env_idx in range(buffer.num_envs)]

        real_tokens = sum(end - start for _, start, end in chunks)
        padded_tokens = max(1, len(chunks) * chunk_len)
        self.writer.add_scalar("charts/sequence_chunks", len(chunks), self.global_step)
        self.writer.add_scalar("charts/sequence_real_token_fraction", real_tokens / padded_tokens, self.global_step)

        normalized_advantages = _normalize(buffer.advantages)
        total_batches = n_epochs * max(1, (len(chunks) + batch_chunks - 1) // batch_chunks)
        batch_counter = 0

        for _ in range(n_epochs):
            order = np.random.permutation(len(chunks))
            for batch_start in range(0, len(order), batch_chunks):
                selected = [chunks[i] for i in order[batch_start : batch_start + batch_chunks]]
                if not selected:
                    continue

                (
                    obs_seq,
                    dir_seq,
                    prev_act_seq,
                    prev_rew_seq,
                    ep_start_seq,
                    action_seq,
                    old_logprob_seq,
                    adv_seq,
                    ret_seq,
                    old_value_seq,
                    loss_mask,
                ) = _pack_sequence_batch(buffer, selected, normalized_advantages, chunk_len)

                logits, new_values = self.model.forward(
                    torch.as_tensor(obs_seq, device=self.device),
                    torch.as_tensor(dir_seq, device=self.device),
                    torch.as_tensor(prev_act_seq, device=self.device),
                    torch.as_tensor(prev_rew_seq, device=self.device),
                    torch.as_tensor(ep_start_seq, device=self.device),
                )
                dist = _safe_categorical(logits)
                new_logprob = dist.log_prob(torch.as_tensor(action_seq, device=self.device).long())
                entropy = dist.entropy()

                stop = self._update_minibatch(
                    new_logprob=new_logprob,
                    entropy=entropy,
                    new_value=new_values,
                    old_logprob=torch.as_tensor(old_logprob_seq, device=self.device),
                    advantages=torch.as_tensor(adv_seq, device=self.device),
                    returns=torch.as_tensor(ret_seq, device=self.device),
                    old_values=torch.as_tensor(old_value_seq, device=self.device),
                    loss_mask=torch.as_tensor(loss_mask, device=self.device),
                )
                batch_counter += 1
                if progress_callback is not None:
                    progress_callback(batch_counter, total_batches)
                if stop:
                    return

    def _update_minibatch(
        self,
        *,
        new_logprob: torch.Tensor,
        entropy: torch.Tensor,
        new_value: torch.Tensor,
        old_logprob: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
    ) -> bool:
        if loss_mask is not None:
            loss_mask = loss_mask.float()
            denom = loss_mask.sum().clamp_min(1.0)

            def reduce_loss(x: torch.Tensor) -> torch.Tensor:
                return (x * loss_mask).sum() / denom

        else:

            def reduce_loss(x: torch.Tensor) -> torch.Tensor:
                return x.mean()

        logratio = new_logprob - old_logprob
        ratio = logratio.exp()

        with torch.no_grad():
            approx_kl = reduce_loss((ratio - 1.0) - logratio)

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
        pg_loss = reduce_loss(torch.max(pg_loss1, pg_loss2))

        value_pred_clipped = old_values + torch.clamp(
            new_value - old_values,
            -self.clip_coef,
            self.clip_coef,
        )
        value_losses = (new_value - returns) ** 2
        value_losses_clipped = (value_pred_clipped - returns) ** 2
        value_loss = 0.5 * reduce_loss(torch.max(value_losses, value_losses_clipped))

        entropy_loss = reduce_loss(entropy)
        loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * value_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.writer.add_scalar("loss/policy", pg_loss.item(), self.global_step)
        self.writer.add_scalar("loss/value", value_loss.item(), self.global_step)
        self.writer.add_scalar("loss/entropy", entropy_loss.item(), self.global_step)
        self.writer.add_scalar("loss/total", loss.item(), self.global_step)
        self.writer.add_scalar("loss/approx_kl", approx_kl.item(), self.global_step)
        self.global_step += 1

        return self.target_kl is not None and approx_kl.item() > self.target_kl


def _normalize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-8)


def _episode_bounded_chunks(buffer: RolloutBuffer, chunk_len: int) -> list[tuple[int, int, int]]:
    """Build chunks that do not cross terminal transitions.

    Short episode fragments are kept and padded later, so sequence PPO does not
    throw away most data when episodes are shorter than the target chunk length.
    """

    chunks: list[tuple[int, int, int]] = []
    for env_idx in range(buffer.num_envs):
        segment_start = 0
        for step in range(buffer.num_steps):
            segment_end = step + 1
            if buffer.dones[step, env_idx] or step == buffer.num_steps - 1:
                start = segment_start
                while start < segment_end:
                    end = min(start + chunk_len, segment_end)
                    chunks.append((env_idx, start, end))
                    start = end
                segment_start = segment_end
    return chunks


def _pack_sequence_batch(
    buffer: RolloutBuffer,
    selected: list[tuple[int, int, int]],
    normalized_advantages: np.ndarray,
    chunk_len: int,
) -> tuple[np.ndarray, ...]:
    batch_size = len(selected)
    obs_shape = buffer.observations.shape[2:]

    obs_seq = np.zeros((batch_size, chunk_len, *obs_shape), dtype=buffer.observations.dtype)
    dir_seq = np.zeros((batch_size, chunk_len, 1), dtype=buffer.directions.dtype)
    prev_act_seq = np.zeros((batch_size, chunk_len, buffer.action_dim), dtype=buffer.prev_actions.dtype)
    prev_rew_seq = np.zeros((batch_size, chunk_len, 1), dtype=buffer.prev_rewards.dtype)
    ep_start_seq = np.zeros((batch_size, chunk_len, 1), dtype=buffer.episode_starts.dtype)
    action_seq = np.zeros((batch_size, chunk_len), dtype=buffer.actions.dtype)
    old_logprob_seq = np.zeros((batch_size, chunk_len), dtype=buffer.logprobs.dtype)
    adv_seq = np.zeros((batch_size, chunk_len), dtype=normalized_advantages.dtype)
    ret_seq = np.zeros((batch_size, chunk_len), dtype=buffer.returns.dtype)
    old_value_seq = np.zeros((batch_size, chunk_len), dtype=buffer.values.dtype)
    loss_mask = np.zeros((batch_size, chunk_len), dtype=np.float32)

    for batch_idx, (env_idx, start, end) in enumerate(selected):
        length = end - start
        if length <= 0:
            continue
        dest = slice(0, length)
        src = slice(start, end)
        obs_seq[batch_idx, dest] = buffer.observations[src, env_idx]
        dir_seq[batch_idx, dest] = buffer.directions[src, env_idx]
        prev_act_seq[batch_idx, dest] = buffer.prev_actions[src, env_idx]
        prev_rew_seq[batch_idx, dest] = buffer.prev_rewards[src, env_idx]
        ep_start_seq[batch_idx, dest] = buffer.episode_starts[src, env_idx]
        action_seq[batch_idx, dest] = buffer.actions[src, env_idx]
        old_logprob_seq[batch_idx, dest] = buffer.logprobs[src, env_idx]
        adv_seq[batch_idx, dest] = normalized_advantages[src, env_idx]
        ret_seq[batch_idx, dest] = buffer.returns[src, env_idx]
        old_value_seq[batch_idx, dest] = buffer.values[src, env_idx]
        loss_mask[batch_idx, dest] = 1.0

    return (
        obs_seq,
        dir_seq,
        prev_act_seq,
        prev_rew_seq,
        ep_start_seq,
        action_seq,
        old_logprob_seq,
        adv_seq,
        ret_seq,
        old_value_seq,
        loss_mask,
    )
