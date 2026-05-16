from __future__ import annotations

import pytest
import torch
import numpy as np

from src.cue import CUE_IGNORE_INDEX, extract_cue_targets_np
from src.impala import vtrace_from_importance_weights
from src.models import EpisodicCueMemory, FastGatedAttentionActorCritic, GRUActorCritic, TokenEncoder
from src.r2d2 import RecurrentSequenceReplay
from src.train_mamba_ppo import Config, _memory_diagnostics, _runtime_metadata


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


def test_slot_memory_checkpoint_metadata_and_strict_roundtrip():
    config = Config(model="gru", d_model=32, slot_count=2, num_envs=1, num_steps=2)
    model = GRUActorCritic(
        action_dim=7,
        d_model=32,
        slot_count=2,
        slot_extractor=config.slot_extractor,
        slot_iters=config.slot_iters,
        temporal_token_mode=config.temporal_token_mode,
        memory_kind=config.memory_kind,
        memory_slots=4,
        memory_topk=2,
        aux_recall=True,
    )
    clone = GRUActorCritic(
        action_dim=7,
        d_model=32,
        slot_count=2,
        slot_extractor=config.slot_extractor,
        slot_iters=config.slot_iters,
        temporal_token_mode=config.temporal_token_mode,
        memory_kind=config.memory_kind,
        memory_slots=4,
        memory_topk=2,
        aux_recall=True,
    )
    missing, unexpected = clone.load_state_dict(model.state_dict(), strict=True)
    assert missing == []
    assert unexpected == []
    metadata = _runtime_metadata(config, action_dim=7)
    assert metadata["config"]["slot_extractor"] == "iterative"
    assert metadata["config"]["temporal_token_mode"] == "fuse"
    assert metadata["config"]["memory_kind"] == "episodic_cue"
    assert metadata["config"]["aux_recall_coef"] == 0.05


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


def _sequence_inputs(batch: int = 2, seq_len: int = 5, action_dim: int = 7):
    obs = torch.zeros(batch, seq_len, 7, 7, 3, dtype=torch.uint8)
    direction = torch.zeros(batch, seq_len, 1, dtype=torch.long)
    prev_action = torch.zeros(batch, seq_len, action_dim)
    prev_reward = torch.zeros(batch, seq_len, 1)
    episode_start = torch.zeros(batch, seq_len, 1)
    valid_mask = torch.ones(batch, seq_len)
    episode_start[:, 0, 0] = 1.0
    obs[:, 0, 3, 3, 0] = 5
    obs[:, 0, 3, 3, 1] = 2
    prev_action[:, :, 0] = 1.0
    return obs, direction, prev_action, prev_reward, episode_start, valid_mask


def test_iterative_slots_and_fused_tokens_forward_smoke():
    obs, direction, prev_action, prev_reward, episode_start, valid_mask = _sequence_inputs()
    encoder = TokenEncoder(
        action_dim=7,
        d_model=32,
        spatial_heads=4,
        slot_count=4,
        slot_extractor="iterative",
        slot_iters=3,
        temporal_token_mode="fuse",
        memory_kind="episodic_cue",
    )
    tokens, token_mask = encoder(
        obs,
        direction,
        prev_action,
        prev_reward,
        episode_start,
        valid_mask=valid_mask,
        return_valid_mask=True,
    )
    assert tokens.shape == (2, 5, 32)
    assert token_mask.shape == (2, 5)
    assert torch.isfinite(tokens).all()


def test_memory_diagnostics_expose_gate_entropy_and_write_rate():
    obs, direction, prev_action, prev_reward, episode_start, valid_mask = _sequence_inputs(batch=1, seq_len=3)
    model = GRUActorCritic(
        action_dim=7,
        d_model=32,
        slot_count=2,
        slot_extractor="iterative",
        temporal_token_mode="fuse",
        memory_kind="episodic_cue",
        aux_recall=True,
    )
    with torch.no_grad():
        model(obs, direction, prev_action, prev_reward, episode_start, valid_mask=valid_mask, return_aux=True)
    diagnostics = _memory_diagnostics(model)
    assert set(diagnostics) == {"gate_mean", "retrieval_entropy", "write_rate"}
    assert 0.0 <= diagnostics["gate_mean"] <= 1.0
    assert diagnostics["retrieval_entropy"] >= 0.0
    assert diagnostics["write_rate"] > 0.0


def test_flatten_and_fuse_modes_preserve_policy_shapes():
    obs, direction, prev_action, prev_reward, episode_start, valid_mask = _sequence_inputs(batch=1, seq_len=4)
    for mode, expected_tokens in [("flatten", 3), ("fuse", 1)]:
        model = GRUActorCritic(
            action_dim=7,
            d_model=32,
            slot_count=2,
            slot_extractor="iterative",
            temporal_token_mode=mode,
            memory_kind="episodic_cue",
            aux_recall=True,
        )
        out = model(obs, direction, prev_action, prev_reward, episode_start, valid_mask=valid_mask, return_aux=True)
        logits, values, aux_logits = out
        assert model.token_encoder.tokens_per_step == expected_tokens
        assert logits.shape == (1, 4, 7)
        assert values.shape == (1, 4)
        assert aux_logits.shape == (1, 4, 512)


def test_episode_cue_memory_resets_on_episode_start():
    memory = EpisodicCueMemory(d_model=16, memory_slots=4, topk=2)
    state = memory.init_state(1, device=torch.device("cpu"), dtype=torch.float32)
    token = torch.randn(1, 16)
    _, state, _, _, write_rate = memory.forward_step(
        token,
        torch.tensor([5 * 16 + 2]),
        torch.ones(1, 1),
        state,
    )
    assert state["valid"].any()
    assert write_rate.item() == 1.0
    _, state, _, _, write_rate = memory.forward_step(
        token,
        torch.tensor([CUE_IGNORE_INDEX]),
        torch.ones(1, 1),
        state,
    )
    assert not state["valid"].any()
    assert write_rate.item() == 0.0


def test_cue_target_extraction_uses_only_visible_non_background_objects():
    obs = np.zeros((2, 7, 7, 3), dtype=np.uint8)
    obs[0, :, :, 0] = 1
    obs[0, 2, 2, 0] = 5
    obs[0, 2, 2, 1] = 3
    obs[1, :, :, 0] = 2
    targets = extract_cue_targets_np(obs)
    assert targets.tolist() == [5 * 16 + 3, CUE_IGNORE_INDEX]


def test_memory_aux_outputs_ignore_padded_tokens():
    torch.manual_seed(0)
    obs, direction, prev_action, prev_reward, episode_start, valid_mask = _sequence_inputs(batch=2, seq_len=6)
    valid_mask[:, 3:] = 0
    noisy_obs = obs.clone()
    noisy_obs[1, 3:, :, :, 0] = torch.randint(4, 8, noisy_obs[1, 3:, :, :, 0].shape, dtype=torch.uint8)
    model = GRUActorCritic(
        action_dim=7,
        d_model=32,
        slot_count=2,
        slot_extractor="iterative",
        temporal_token_mode="fuse",
        memory_kind="episodic_cue",
        aux_recall=True,
    )
    model.eval()
    with torch.no_grad():
        logits, values, aux_logits = model(
            noisy_obs,
            direction,
            prev_action,
            prev_reward,
            episode_start,
            valid_mask=valid_mask,
            return_aux=True,
        )
    torch.testing.assert_close(logits[0, :3], logits[1, :3], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(values[0, :3], values[1, :3], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(aux_logits[0, :3], aux_logits[1, :3], atol=1e-5, rtol=1e-5)


def test_gated_attention_stateful_matches_full_context_with_memory():
    torch.manual_seed(0)
    obs, direction, prev_action, prev_reward, episode_start, valid_mask = _sequence_inputs(batch=1, seq_len=3)
    model = FastGatedAttentionActorCritic(
        action_dim=7,
        d_model=32,
        n_layers=1,
        n_heads=4,
        context_len=3,
        position_mode="alibi",
        slot_count=2,
        temporal_token_mode="fuse",
        memory_kind="episodic_cue",
    )
    model.eval()
    with torch.no_grad():
        full_logits, full_values = model(obs, direction, prev_action, prev_reward, episode_start, valid_mask=valid_mask)
        state = model.init_inference_state(1, device=torch.device("cpu"), dtype=torch.float32)
        step_logprobs = []
        step_values = []
        for idx in range(3):
            action, _logp, _entropy, value, state = model.get_action_and_value_step(
                obs[:, idx],
                direction[:, idx],
                prev_action[:, idx],
                prev_reward[:, idx],
                episode_start[:, idx],
                state,
                action=torch.zeros(1, dtype=torch.long),
            )
            del action
            step_values.append(value)
            step_logprobs.append(_logp)
        assert torch.isfinite(torch.stack(step_values)).all()
        step_values = torch.stack(step_values, dim=1)
        step_logprobs = torch.stack(step_logprobs, dim=1)
        full_logprobs = torch.distributions.Categorical(logits=full_logits).log_prob(torch.zeros(1, 3, dtype=torch.long))
        torch.testing.assert_close(step_values, full_values, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(step_logprobs, full_logprobs, atol=1e-5, rtol=1e-5)
    assert full_logits.shape == (1, 3, 7)
    assert full_values.shape == (1, 3)
