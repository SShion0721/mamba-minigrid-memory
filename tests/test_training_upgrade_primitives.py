from __future__ import annotations

import pytest
import torch
import numpy as np

from src.impala import vtrace_from_importance_weights
from src.models import GRUActorCritic
from src.r2d2 import RecurrentSequenceReplay


def test_checkpoint_strict_rejects_unexpected_keys_but_legacy_allows():
    model = GRUActorCritic(action_dim=7, d_model=32)
    state = model.state_dict()
    legacy_state = dict(state)
    legacy_state["token_encoder.legacy_gate.weight"] = torch.zeros(1)

    with pytest.raises(RuntimeError):
        model.load_state_dict(legacy_state, strict=True)

    missing, unexpected = model.load_state_dict(legacy_state, strict=False)
    assert not missing
    assert unexpected == ["token_encoder.legacy_gate.weight"]


def test_vtrace_matches_td_when_on_policy_and_unclipped():
    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.tensor([[0.5], [0.25]])
    bootstrap = torch.tensor([0.0])
    discounts = torch.tensor([[0.9], [0.9]])
    out = vtrace_from_importance_weights(
        log_rhos=torch.zeros_like(rewards),
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap,
    )

    expected_last = rewards[1] + discounts[1] * bootstrap
    expected_first = rewards[0] + discounts[0] * expected_last
    torch.testing.assert_close(out.values[:, 0], torch.stack([expected_first[0], expected_last[0]]))
    assert out.pg_advantages.shape == rewards.shape


def test_recurrent_sequence_replay_samples_and_updates_priorities():
    replay = RecurrentSequenceReplay(capacity=4, sequence_len=5, burn_in_len=2, seed=123)
    for idx in range(3):
        replay.add({"obs": np.full((5, 2), idx, dtype=np.float32)}, priority=idx + 1)

    batch, indices, weights = replay.sample(batch_size=2)
    assert len(batch) == 2
    assert indices.shape == (2,)
    assert weights.shape == (2,)

    replay.update_priorities(indices, np.full_like(indices, 10, dtype=np.float32))
    assert len(replay) == 3
