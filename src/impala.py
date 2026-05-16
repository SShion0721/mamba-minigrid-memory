"""IMPALA/V-trace utilities for recurrent MiniGrid experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VTraceTargets:
    """V-trace value targets and policy-gradient advantages."""

    values: torch.Tensor
    pg_advantages: torch.Tensor


def vtrace_from_importance_weights(
    *,
    log_rhos: torch.Tensor,
    discounts: torch.Tensor,
    rewards: torch.Tensor,
    values: torch.Tensor,
    bootstrap_value: torch.Tensor,
    clip_rho_threshold: float = 1.0,
    clip_pg_rho_threshold: float = 1.0,
    clip_c_threshold: float = 1.0,
) -> VTraceTargets:
    """Compute V-trace targets.

    Tensor shapes are time-major: ``[T, B]`` for sequence tensors and ``[B]``
    for ``bootstrap_value``. ``discounts`` should already include gamma and
    terminal masking.
    """

    if log_rhos.shape != rewards.shape or rewards.shape != values.shape:
        raise ValueError("log_rhos, rewards, and values must have shape [T, B].")
    if discounts.shape != rewards.shape:
        raise ValueError("discounts must have shape [T, B].")
    if bootstrap_value.shape != rewards.shape[1:]:
        raise ValueError("bootstrap_value must have shape [B].")

    rhos = torch.exp(log_rhos)
    clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
    clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
    cs = torch.clamp(rhos, max=clip_c_threshold)

    values_t_plus_1 = torch.cat([values[1:], bootstrap_value.unsqueeze(0)], dim=0)
    deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

    acc = torch.zeros_like(bootstrap_value)
    vs_minus_values = []
    for t in reversed(range(rewards.shape[0])):
        acc = deltas[t] + discounts[t] * cs[t] * acc
        vs_minus_values.append(acc)
    vs_minus_values = torch.stack(list(reversed(vs_minus_values)), dim=0)
    vs = values + vs_minus_values

    vs_t_plus_1 = torch.cat([vs[1:], bootstrap_value.unsqueeze(0)], dim=0)
    pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values)
    return VTraceTargets(values=vs, pg_advantages=pg_advantages)
