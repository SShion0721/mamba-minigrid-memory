"""Rollout storage and PPO updates."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.cue import CUE_IGNORE_INDEX, extract_cue_targets_np
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
        if context_len <= 0:
            raise ValueError("context_len must be positive.")
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.context_len = context_len
        self.action_dim = action_dim

        self.observations = np.zeros((num_steps, num_envs, *obs_shape), dtype=np.uint8)
        self.directions = np.zeros((num_steps, num_envs, 1), dtype=np.int64)
        self.actions = np.zeros((num_steps, num_envs), dtype=np.int64)
        self.prev_actions = np.zeros((num_steps, num_envs, action_dim), dtype=np.float32)
        self.prev_rewards = np.zeros((num_steps, num_envs, 1), dtype=np.float32)
        self.episode_starts = np.zeros((num_steps, num_envs, 1), dtype=np.float32)
        self.logprobs = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_envs), dtype=bool)
        self.cue_targets = np.full((num_steps, num_envs), CUE_IGNORE_INDEX, dtype=np.int64)
        self.episode_cues = np.full(num_envs, CUE_IGNORE_INDEX, dtype=np.int64)

        self.advantages: np.ndarray | None = None
        self.returns: np.ndarray | None = None

        self.context_obs = np.zeros((num_envs, context_len, *obs_shape), dtype=self.observations.dtype)
        self.context_dirs = np.zeros((num_envs, context_len, 1), dtype=self.directions.dtype)
        self.context_actions = np.zeros((num_envs, context_len, action_dim), dtype=self.prev_actions.dtype)
        self.context_rewards = np.zeros((num_envs, context_len, 1), dtype=self.prev_rewards.dtype)
        self.context_starts = np.zeros((num_envs, context_len, 1), dtype=self.episode_starts.dtype)
        self.context_valid = np.zeros((num_envs, context_len), dtype=np.float32)

        self.hist_obs = np.zeros_like(self.context_obs)
        self.hist_dirs = np.zeros_like(self.context_dirs)
        self.hist_actions = np.zeros_like(self.context_actions)
        self.hist_rewards = np.zeros_like(self.context_rewards)
        self.hist_starts = np.zeros_like(self.context_starts)
        self.hist_pos = np.zeros(num_envs, dtype=np.int64)
        self.hist_len = np.zeros(num_envs, dtype=np.int64)
        self.env_indices = np.arange(num_envs)

        self.step = 0

    def reset_context(self, env_idx: int) -> None:
        self.hist_pos[env_idx] = 0
        self.hist_len[env_idx] = 0
        self.episode_cues[env_idx] = CUE_IGNORE_INDEX

    def reset_contexts(self, done_mask: np.ndarray) -> None:
        if done_mask.any():
            self.hist_pos[done_mask] = 0
            self.hist_len[done_mask] = 0
            self.episode_cues[done_mask] = CUE_IGNORE_INDEX

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
        self._update_episode_cues(idx, np.asarray(obs)[None], np.asarray(episode_start)[None], np.asarray([env_idx]))

        pos = self.hist_pos[env_idx]
        self.hist_obs[env_idx, pos] = obs
        self.hist_dirs[env_idx, pos] = direction
        self.hist_actions[env_idx, pos] = prev_action
        self.hist_rewards[env_idx, pos] = prev_reward
        self.hist_starts[env_idx, pos] = episode_start
        self.hist_pos[env_idx] = (pos + 1) % self.context_len
        self.hist_len[env_idx] = min(self.hist_len[env_idx] + 1, self.context_len)

        if done:
            self.reset_context(env_idx)

    def add_batch(
        self,
        *,
        obs: np.ndarray,
        directions: np.ndarray,
        actions: np.ndarray,
        logprobs: np.ndarray,
        values: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        prev_actions: np.ndarray,
        prev_rewards: np.ndarray,
        episode_starts: np.ndarray,
    ) -> None:
        idx = self.step
        self.observations[idx] = obs
        self.directions[idx] = directions
        self.actions[idx] = actions
        self.prev_actions[idx] = prev_actions
        self.prev_rewards[idx] = prev_rewards
        self.episode_starts[idx] = episode_starts
        self.logprobs[idx] = logprobs
        self.rewards[idx] = rewards
        self.values[idx] = values
        self.dones[idx] = dones
        self._update_episode_cues(idx, obs, episode_starts, self.env_indices)

        cols = self.hist_pos
        self.hist_obs[self.env_indices, cols] = obs
        self.hist_dirs[self.env_indices, cols] = directions
        self.hist_actions[self.env_indices, cols] = prev_actions
        self.hist_rewards[self.env_indices, cols] = prev_rewards
        self.hist_starts[self.env_indices, cols] = episode_starts
        self.hist_pos = (self.hist_pos + 1) % self.context_len
        self.hist_len = np.minimum(self.hist_len + 1, self.context_len)
        self.reset_contexts(dones)

    def _update_episode_cues(
        self,
        step_idx: int,
        obs: np.ndarray,
        episode_starts: np.ndarray,
        env_indices: np.ndarray,
    ) -> None:
        starts = np.asarray(episode_starts).reshape(len(env_indices), -1)[:, 0] > 0.5
        if starts.any():
            self.episode_cues[env_indices[starts]] = CUE_IGNORE_INDEX
        visible_cues = extract_cue_targets_np(obs)
        missing = self.episode_cues[env_indices] == CUE_IGNORE_INDEX
        has_visible = visible_cues != CUE_IGNORE_INDEX
        write = missing & has_visible
        if write.any():
            self.episode_cues[env_indices[write]] = visible_cues[write]
        self.cue_targets[step_idx, env_indices] = self.episode_cues[env_indices]

    def get_context(
        self,
        current_obs: np.ndarray | None = None,
        current_directions: np.ndarray | None = None,
        current_prev_actions: np.ndarray | None = None,
        current_prev_rewards: np.ndarray | None = None,
        current_episode_starts: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return left-aligned context windows, optionally including current token."""

        ctx_len = self.context_len
        obs_ctx = self.context_obs
        dir_ctx = self.context_dirs
        act_ctx = self.context_actions
        rew_ctx = self.context_rewards
        start_ctx = self.context_starts
        valid_ctx = self.context_valid
        obs_ctx.fill(0)
        dir_ctx.fill(0)
        act_ctx.fill(0)
        rew_ctx.fill(0)
        start_ctx.fill(0)
        valid_ctx.fill(0)

        include_current = current_obs is not None
        hist_capacity = ctx_len - int(include_current)
        for env_idx in range(self.num_envs):
            hist_take = min(int(self.hist_len[env_idx]), hist_capacity)
            if hist_take:
                dest = slice(0, hist_take)
                _copy_ring_tail(self.hist_obs[env_idx], self.hist_pos[env_idx], hist_take, obs_ctx[env_idx, dest])
                _copy_ring_tail(self.hist_dirs[env_idx], self.hist_pos[env_idx], hist_take, dir_ctx[env_idx, dest])
                _copy_ring_tail(
                    self.hist_actions[env_idx],
                    self.hist_pos[env_idx],
                    hist_take,
                    act_ctx[env_idx, dest],
                )
                _copy_ring_tail(self.hist_rewards[env_idx], self.hist_pos[env_idx], hist_take, rew_ctx[env_idx, dest])
                _copy_ring_tail(self.hist_starts[env_idx], self.hist_pos[env_idx], hist_take, start_ctx[env_idx, dest])
                valid_ctx[env_idx, dest] = 1.0

        if include_current:
            insert_at = np.minimum(self.hist_len, hist_capacity)
            obs_ctx[self.env_indices, insert_at] = current_obs
            dir_ctx[self.env_indices, insert_at] = current_directions
            act_ctx[self.env_indices, insert_at] = current_prev_actions
            rew_ctx[self.env_indices, insert_at] = current_prev_rewards
            start_ctx[self.env_indices, insert_at] = current_episode_starts
            valid_ctx[self.env_indices, insert_at] = 1.0

        return obs_ctx, dir_ctx, act_ctx, rew_ctx, start_ctx, valid_ctx

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


def _copy_ring_tail(history: np.ndarray, next_pos: int, count: int, dest: np.ndarray) -> None:
    start = (int(next_pos) - count) % history.shape[0]
    end = start + count
    if end <= history.shape[0]:
        dest[...] = history[start:end]
        return

    first = history.shape[0] - start
    dest[:first] = history[start:]
    dest[first:] = history[: count - first]


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
        clip_vloss: bool = False,
        norm_adv: bool = True,
        amp_dtype: torch.dtype | None = None,
        aux_recall_coef: float = 0.0,
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
        self.clip_vloss = clip_vloss
        self.norm_adv = norm_adv
        self.amp_dtype = amp_dtype if device.type == "cuda" else None
        self.aux_recall_coef = float(aux_recall_coef)
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.amp_dtype == torch.float16)
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
        b_advantages = buffer.advantages.reshape(-1)
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

                with self._autocast():
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
        dynamic_sequence_len: bool = False,
    ) -> None:
        self.train_sequence_with_callback(
            buffer,
            chunk_len=chunk_len,
            batch_chunks=batch_chunks,
            n_epochs=n_epochs,
            dynamic_sequence_len=dynamic_sequence_len,
            progress_callback=None,
        )

    def train_sequence_with_callback(
        self,
        buffer: RolloutBuffer,
        chunk_len: int,
        batch_chunks: int,
        n_epochs: int,
        dynamic_sequence_len: bool = False,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        if buffer.advantages is None or buffer.returns is None:
            raise RuntimeError("Call buffer.compute_gae(...) before PPO updates.")

        chunk_len = min(chunk_len, buffer.num_steps)
        burn_in_len = max(0, buffer.context_len - chunk_len)
        sequence_len = chunk_len + burn_in_len
        chunks = _episode_bounded_chunks(buffer, chunk_len)
        if not chunks:
            chunks = [(env_idx, 0, 0, buffer.num_steps) for env_idx in range(buffer.num_envs)]

        real_tokens = sum(end - max(segment_start, start - burn_in_len) for _, segment_start, start, end in chunks)
        target_tokens = sum(end - start for _, _, start, end in chunks)
        padded_tokens = max(1, len(chunks) * sequence_len)
        self.writer.add_scalar("charts/sequence_chunks", len(chunks), self.global_step)
        self.writer.add_scalar("charts/sequence_real_token_fraction", real_tokens / padded_tokens, self.global_step)
        self.writer.add_scalar("charts/sequence_target_token_fraction", target_tokens / padded_tokens, self.global_step)

        advantages = buffer.advantages
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
                    valid_mask,
                    loss_mask,
                    lengths,
                    cue_target_seq,
                ) = _pack_sequence_batch(
                    buffer,
                    selected,
                    advantages,
                    chunk_len,
                    burn_in_len,
                    fixed_sequence_len=not dynamic_sequence_len,
                )

                model_valid_mask = None if bool(valid_mask.all()) else valid_mask
                model_lengths = None if model_valid_mask is None else lengths
                update_loss_mask = None if bool(loss_mask.all()) else loss_mask

                with self._autocast():
                    model_out = self.model.forward(
                        torch.as_tensor(obs_seq, device=self.device),
                        torch.as_tensor(dir_seq, device=self.device),
                        torch.as_tensor(prev_act_seq, device=self.device),
                        torch.as_tensor(prev_rew_seq, device=self.device),
                        torch.as_tensor(ep_start_seq, device=self.device),
                        valid_mask=(
                            None
                            if model_valid_mask is None
                            else torch.as_tensor(model_valid_mask, device=self.device)
                        ),
                        lengths=(
                            None
                            if model_lengths is None
                            else torch.as_tensor(model_lengths, device=self.device)
                        ),
                        return_aux=self.aux_recall_coef > 0,
                    )
                    if self.aux_recall_coef > 0 and len(model_out) == 3:
                        logits, new_values, aux_logits = model_out
                    else:
                        logits, new_values = model_out[:2]
                        aux_logits = None
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
                    loss_mask=(
                        None
                        if update_loss_mask is None
                        else torch.as_tensor(update_loss_mask, device=self.device)
                    ),
                    aux_logits=aux_logits,
                    cue_targets=torch.as_tensor(cue_target_seq, device=self.device).long(),
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
        aux_logits: torch.Tensor | None = None,
        cue_targets: torch.Tensor | None = None,
    ) -> bool:
        if loss_mask is not None:
            loss_mask = loss_mask.float()
            denom = loss_mask.sum().clamp_min(1.0)

            def reduce_loss(x: torch.Tensor) -> torch.Tensor:
                return (x * loss_mask).sum() / denom

        else:

            def reduce_loss(x: torch.Tensor) -> torch.Tensor:
                return x.mean()

        if self.norm_adv:
            with torch.no_grad():
                if loss_mask is not None:
                    adv_mean = reduce_loss(advantages)
                    adv_var = reduce_loss((advantages - adv_mean) ** 2)
                    advantages = (advantages - adv_mean) / torch.sqrt(adv_var + 1e-8)
                else:
                    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        logratio = new_logprob - old_logprob
        ratio = logratio.exp()

        with torch.no_grad():
            old_approx_kl = reduce_loss(-logratio)
            approx_kl = reduce_loss((ratio - 1.0) - logratio)
            clipfrac = reduce_loss(((ratio - 1.0).abs() > self.clip_coef).float())

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
        pg_loss = reduce_loss(torch.max(pg_loss1, pg_loss2))

        if self.clip_vloss:
            value_loss_unclipped = (new_value - returns) ** 2
            value_clipped = old_values + torch.clamp(new_value - old_values, -self.clip_coef, self.clip_coef)
            value_loss_clipped = (value_clipped - returns) ** 2
            value_loss = 0.5 * reduce_loss(torch.max(value_loss_unclipped, value_loss_clipped))
        else:
            value_loss = 0.5 * reduce_loss((new_value - returns) ** 2)

        entropy_loss = reduce_loss(entropy)
        loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * value_loss
        aux_loss = torch.zeros((), device=new_value.device)
        aux_accuracy = torch.full((), float("nan"), device=new_value.device)
        if self.aux_recall_coef > 0 and aux_logits is not None and cue_targets is not None:
            cue_targets = cue_targets.to(device=aux_logits.device).long()
            cue_keep = cue_targets != CUE_IGNORE_INDEX
            if loss_mask is not None:
                cue_keep = cue_keep & loss_mask.bool().to(device=aux_logits.device)
            if cue_keep.any():
                aux_loss = F.cross_entropy(aux_logits[cue_keep], cue_targets[cue_keep])
                aux_pred = aux_logits[cue_keep].argmax(dim=-1)
                aux_accuracy = (aux_pred == cue_targets[cue_keep]).float().mean()
                loss = loss + self.aux_recall_coef * aux_loss

        self.optimizer.zero_grad(set_to_none=True)
        if self.grad_scaler.is_enabled():
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

        with torch.no_grad():
            explained_var = _explained_variance(new_value.detach(), returns.detach(), loss_mask)

        self.writer.add_scalar("loss/policy", pg_loss.item(), self.global_step)
        self.writer.add_scalar("loss/value", value_loss.item(), self.global_step)
        self.writer.add_scalar("loss/entropy", entropy_loss.item(), self.global_step)
        self.writer.add_scalar("loss/total", loss.item(), self.global_step)
        self.writer.add_scalar("loss/aux_recall", aux_loss.item(), self.global_step)
        if bool(torch.isfinite(aux_accuracy).item()):
            self.writer.add_scalar("loss/aux_recall_acc", aux_accuracy.item(), self.global_step)
        self.writer.add_scalar("loss/old_approx_kl", old_approx_kl.item(), self.global_step)
        self.writer.add_scalar("loss/approx_kl", approx_kl.item(), self.global_step)
        self.writer.add_scalar("loss/clipfrac", clipfrac.item(), self.global_step)
        self.writer.add_scalar("loss/explained_variance", explained_var, self.global_step)
        self.global_step += 1

        return self.target_kl is not None and approx_kl.item() > self.target_kl

    def _autocast(self):
        if self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)


def _explained_variance(predicted: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor | None) -> float:
    if loss_mask is not None:
        keep = loss_mask.bool()
        predicted = predicted[keep]
        target = target[keep]
    if target.numel() == 0:
        return float("nan")
    target_var = torch.var(target, unbiased=False)
    if target_var <= 0:
        return float("nan")
    return float((1.0 - torch.var(target - predicted, unbiased=False) / target_var).item())


def _episode_bounded_chunks(buffer: RolloutBuffer, chunk_len: int) -> list[tuple[int, int, int, int]]:
    """Build chunks that do not cross episode boundaries.

    Short episode fragments are kept and padded later, so sequence PPO does not
    throw away most data when episodes are shorter than the target chunk length.
    """

    chunks: list[tuple[int, int, int, int]] = []

    def append_segment(env_idx: int, segment_start: int, segment_end: int) -> None:
        start = segment_start
        while start < segment_end:
            end = min(start + chunk_len, segment_end)
            chunks.append((env_idx, segment_start, start, end))
            start = end

    for env_idx in range(buffer.num_envs):
        segment_start = 0
        for step in range(buffer.num_steps):
            starts_new_episode = (
                step > segment_start
                and buffer.episode_starts[step, env_idx, 0] > 0.5
            )
            if starts_new_episode:
                append_segment(env_idx, segment_start, step)
                segment_start = step

            segment_end = step + 1
            if buffer.dones[step, env_idx] or step == buffer.num_steps - 1:
                append_segment(env_idx, segment_start, segment_end)
                segment_start = segment_end
    return chunks


def _pack_sequence_batch(
    buffer: RolloutBuffer,
    selected: list[tuple[int, int, int, int]],
    advantages: np.ndarray,
    chunk_len: int,
    burn_in_len: int,
    *,
    fixed_sequence_len: bool = True,
) -> tuple[np.ndarray, ...]:
    batch_size = len(selected)
    obs_shape = buffer.observations.shape[2:]
    max_sequence_len = chunk_len + burn_in_len
    if fixed_sequence_len:
        sequence_len = max_sequence_len
    else:
        sequence_len = 1
        for env_idx, segment_start, start, end in selected:
            del env_idx
            target_length = end - start
            if target_length <= 0:
                continue
            burn_start = max(segment_start, start - burn_in_len)
            sequence_len = max(sequence_len, end - burn_start)
        sequence_len = min(sequence_len, max_sequence_len)

    obs_seq = np.zeros((batch_size, sequence_len, *obs_shape), dtype=buffer.observations.dtype)
    dir_seq = np.zeros((batch_size, sequence_len, 1), dtype=buffer.directions.dtype)
    prev_act_seq = np.zeros((batch_size, sequence_len, buffer.action_dim), dtype=buffer.prev_actions.dtype)
    prev_rew_seq = np.zeros((batch_size, sequence_len, 1), dtype=buffer.prev_rewards.dtype)
    ep_start_seq = np.zeros((batch_size, sequence_len, 1), dtype=buffer.episode_starts.dtype)
    action_seq = np.zeros((batch_size, sequence_len), dtype=buffer.actions.dtype)
    old_logprob_seq = np.zeros((batch_size, sequence_len), dtype=buffer.logprobs.dtype)
    adv_seq = np.zeros((batch_size, sequence_len), dtype=advantages.dtype)
    ret_seq = np.zeros((batch_size, sequence_len), dtype=buffer.returns.dtype)
    old_value_seq = np.zeros((batch_size, sequence_len), dtype=buffer.values.dtype)
    cue_target_seq = np.full((batch_size, sequence_len), CUE_IGNORE_INDEX, dtype=np.int64)
    valid_mask = np.zeros((batch_size, sequence_len), dtype=np.float32)
    loss_mask = np.zeros((batch_size, sequence_len), dtype=np.float32)
    lengths = np.zeros(batch_size, dtype=np.int64)

    for batch_idx, (env_idx, segment_start, start, end) in enumerate(selected):
        burn_start = max(segment_start, start - burn_in_len)
        sequence_end = end
        sequence_length = sequence_end - burn_start
        target_start = start - burn_start
        target_length = end - start
        if sequence_length <= 0 or target_length <= 0:
            continue
        offset = 0
        dest = slice(offset, offset + sequence_length)
        src = slice(burn_start, sequence_end)
        obs_seq[batch_idx, dest] = buffer.observations[src, env_idx]
        dir_seq[batch_idx, dest] = buffer.directions[src, env_idx]
        prev_act_seq[batch_idx, dest] = buffer.prev_actions[src, env_idx]
        prev_rew_seq[batch_idx, dest] = buffer.prev_rewards[src, env_idx]
        ep_start_seq[batch_idx, dest] = buffer.episode_starts[src, env_idx]
        action_seq[batch_idx, dest] = buffer.actions[src, env_idx]
        old_logprob_seq[batch_idx, dest] = buffer.logprobs[src, env_idx]
        adv_seq[batch_idx, dest] = advantages[src, env_idx]
        ret_seq[batch_idx, dest] = buffer.returns[src, env_idx]
        old_value_seq[batch_idx, dest] = buffer.values[src, env_idx]
        cue_target_seq[batch_idx, dest] = buffer.cue_targets[src, env_idx]
        valid_mask[batch_idx, dest] = 1.0
        lengths[batch_idx] = sequence_length
        loss_start = offset + target_start
        loss_mask[batch_idx, loss_start : loss_start + target_length] = 1.0

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
        valid_mask,
        loss_mask,
        lengths,
        cue_target_seq,
    )
