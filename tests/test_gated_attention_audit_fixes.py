from __future__ import annotations

from collections import deque
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval import _sequence_action as eval_sequence_action
from src.models import (
    AttentionActorCritic,
    FastGatedAttentionActorCritic,
    LSTMActorCritic,
    MLPActorCritic,
)
from src.ppo import RolloutBuffer, _pack_sequence_batch
from src.visualize import _sequence_action as visualize_sequence_action


def _sample_sequence(batch: int = 2, seq_len: int = 6, action_dim: int = 7):
    obs = torch.zeros(batch, seq_len, 7, 7, 3, dtype=torch.uint8)
    direction = torch.zeros(batch, seq_len, 1, dtype=torch.long)
    prev_action = torch.zeros(batch, seq_len, action_dim)
    prev_reward = torch.zeros(batch, seq_len, 1)
    episode_start = torch.zeros(batch, seq_len, 1)
    valid_mask = torch.zeros(batch, seq_len)
    valid_mask[:, -3:] = 1.0

    obs[:, -3:, 3, 3, 0] = torch.tensor([2, 3, 4], dtype=torch.uint8)
    obs[:, -3:, 3, 3, 1] = torch.tensor([1, 2, 3], dtype=torch.uint8)
    direction[:, -3:, 0] = torch.tensor([0, 1, 2], dtype=torch.long)
    prev_action[:, -3:, 0] = 1.0
    prev_reward[:, -3:, 0] = torch.tensor([0.0, 0.25, 0.5])
    episode_start[:, -3:, 0] = torch.tensor([1.0, 0.0, 0.0])
    return obs, direction, prev_action, prev_reward, episode_start, valid_mask


def test_pack_sequence_batch_right_aligns_valid_and_loss_masks():
    buffer = RolloutBuffer(num_envs=1, num_steps=5, obs_shape=(1, 1, 3), action_dim=3, context_len=8)

    for step in range(buffer.num_steps):
        buffer.observations[step, 0, 0, 0, 0] = step + 1
        buffer.directions[step, 0, 0] = step % 4
        buffer.prev_actions[step, 0, step % 3] = 1.0
        buffer.prev_rewards[step, 0, 0] = step / 10.0
        buffer.episode_starts[step, 0, 0] = 1.0 if step == 0 else 0.0
        buffer.actions[step, 0] = step % 3
        buffer.logprobs[step, 0] = -float(step)
        buffer.values[step, 0] = float(step)

    advantages = np.arange(buffer.num_steps, dtype=np.float32).reshape(buffer.num_steps, 1)
    buffer.returns = advantages + 10.0

    (
        obs_seq,
        _dir_seq,
        _prev_act_seq,
        _prev_rew_seq,
        _ep_start_seq,
        _action_seq,
        _old_logprob_seq,
        _adv_seq,
        _ret_seq,
        _old_value_seq,
        valid_mask,
        loss_mask,
    ) = _pack_sequence_batch(buffer, [(0, 0, 2, 4)], advantages, chunk_len=4, burn_in_len=4)

    assert obs_seq.shape[1] == 8
    assert obs_seq[0, :, 0, 0, 0].tolist() == [0, 0, 0, 0, 1, 2, 3, 4]
    assert valid_mask[0].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert loss_mask[0].tolist() == [0, 0, 0, 0, 0, 0, 1, 1]


def test_gated_attention_valid_mask_ignores_padding_tokens():
    for position_mode in ["learned", "alibi"]:
        torch.manual_seed(0)
        model = FastGatedAttentionActorCritic(
            action_dim=7,
            d_model=32,
            n_layers=2,
            n_heads=4,
            context_len=6,
            position_mode=position_mode,
        )
        model.eval()

        obs, direction, prev_action, prev_reward, episode_start, valid_mask = _sample_sequence()
        noisy_obs = obs.clone()
        noisy_direction = direction.clone()
        noisy_prev_action = prev_action.clone()
        noisy_prev_reward = prev_reward.clone()

        noisy_obs[1, :3] = torch.randint(0, 5, noisy_obs[1, :3].shape, dtype=torch.uint8)
        noisy_direction[1, :3, 0] = torch.tensor([3, 2, 1])
        noisy_prev_action[1, :3, 4] = 1.0
        noisy_prev_reward[1, :3, 0] = torch.tensor([9.0, 8.0, 7.0])

        with torch.no_grad():
            logits, values = model(
                noisy_obs,
                noisy_direction,
                noisy_prev_action,
                noisy_prev_reward,
                episode_start,
                valid_mask=valid_mask,
            )

        torch.testing.assert_close(logits[0, -3:], logits[1, -3:], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(values[0, -3:], values[1, -3:], atol=1e-5, rtol=1e-5)


def test_sequence_model_forward_smoke_with_valid_mask():
    obs, direction, prev_action, prev_reward, episode_start, valid_mask = _sample_sequence(batch=1)

    sequence_models = [
        LSTMActorCritic(action_dim=7, d_model=32),
        AttentionActorCritic(action_dim=7, d_model=32, n_heads=4, context_len=6),
        FastGatedAttentionActorCritic(action_dim=7, d_model=32, n_heads=4, context_len=6, position_mode="learned"),
        FastGatedAttentionActorCritic(action_dim=7, d_model=32, n_heads=4, context_len=6, position_mode="alibi"),
    ]

    for model in sequence_models:
        model.eval()
        with torch.no_grad():
            logits, values = model(obs, direction, prev_action, prev_reward, episode_start, valid_mask=valid_mask)
        assert logits.shape == (1, 6, 7)
        assert values.shape == (1, 6)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(values).all()

    mlp = MLPActorCritic(action_dim=7, d_model=32)
    with torch.no_grad():
        logits, values = mlp(obs[:, -1], direction[:, -1], prev_action[:, -1], prev_reward[:, -1], episode_start[:, -1])
    assert logits.shape == (1, 7)
    assert values.shape == (1,)


def test_eval_and_visualize_stochastic_sampling_are_nan_safe():
    class BadLogitModel:
        def forward(self, obs_seq, *_args):
            batch, seq_len = obs_seq.shape[:2]
            logits = torch.zeros(batch, seq_len, 3)
            logits[:, -1] = torch.tensor([float("nan"), float("inf"), float("-inf")])
            return logits, torch.zeros(batch, seq_len)

    obs_ctx = deque([np.zeros((7, 7, 3), dtype=np.uint8)], maxlen=4)
    dir_ctx = deque([np.zeros((1,), dtype=np.int64)], maxlen=4)
    act_ctx = deque([np.zeros((3,), dtype=np.float32)], maxlen=4)
    rew_ctx = deque([np.zeros((1,), dtype=np.float32)], maxlen=4)
    start_ctx = deque([np.ones((1,), dtype=np.float32)], maxlen=4)

    action = eval_sequence_action(
        BadLogitModel(),
        torch.device("cpu"),
        obs_ctx,
        dir_ctx,
        act_ctx,
        rew_ctx,
        start_ctx,
        context_len=4,
        action_dim=3,
        deterministic=False,
    )
    assert action in {0, 1, 2}

    action, probs = visualize_sequence_action(
        BadLogitModel(),
        torch.device("cpu"),
        obs_ctx,
        dir_ctx,
        act_ctx,
        rew_ctx,
        start_ctx,
        context_len=4,
        action_dim=3,
        deterministic=False,
    )
    assert action in {0, 1, 2}
    assert np.isfinite(probs).all()


if __name__ == "__main__":
    for name, test_fn in sorted(globals().items()):
        if name.startswith("test_") and callable(test_fn):
            test_fn()
