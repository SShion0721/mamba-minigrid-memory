"""Cue extraction helpers for MiniGrid Memory observations."""

from __future__ import annotations

import numpy as np
import torch


CUE_IGNORE_INDEX = -100
CUE_CLASS_COUNT = 32 * 16
BACKGROUND_OBJECT_IDS = (0, 1, 2, 3, 10)


def extract_cue_targets_np(obs: np.ndarray) -> np.ndarray:
    """Return the first visible non-background object/color class per frame."""

    obs = np.asarray(obs)
    obj = obs[..., 0].astype(np.int64, copy=False)
    color = obs[..., 1].astype(np.int64, copy=False)
    flat_obj = obj.reshape(*obj.shape[:-2], -1)
    flat_color = color.reshape(*color.shape[:-2], -1)
    valid = np.ones_like(flat_obj, dtype=bool)
    for object_id in BACKGROUND_OBJECT_IDS:
        valid &= flat_obj != object_id

    any_valid = valid.any(axis=-1)
    first_idx = valid.argmax(axis=-1)
    chosen_obj = np.take_along_axis(flat_obj, first_idx[..., None], axis=-1).squeeze(-1)
    chosen_color = np.take_along_axis(flat_color, first_idx[..., None], axis=-1).squeeze(-1)
    target = chosen_obj * 16 + np.clip(chosen_color, 0, 15)
    return np.where(any_valid, target, CUE_IGNORE_INDEX).astype(np.int64, copy=False)


def extract_cue_targets_torch(obs: torch.Tensor) -> torch.Tensor:
    """Torch equivalent of :func:`extract_cue_targets_np`."""

    obj = obs[..., 0].long().clamp(0, 31)
    color = obs[..., 1].long().clamp(0, 15)
    flat_obj = obj.reshape(*obj.shape[:-2], -1)
    flat_color = color.reshape(*color.shape[:-2], -1)
    valid = torch.ones_like(flat_obj, dtype=torch.bool)
    for object_id in BACKGROUND_OBJECT_IDS:
        valid = valid & (flat_obj != object_id)

    any_valid = valid.any(dim=-1)
    first_idx = valid.long().argmax(dim=-1)
    chosen_obj = flat_obj.gather(-1, first_idx.unsqueeze(-1)).squeeze(-1)
    chosen_color = flat_color.gather(-1, first_idx.unsqueeze(-1)).squeeze(-1)
    target = chosen_obj * 16 + chosen_color
    ignore = torch.full_like(target, CUE_IGNORE_INDEX)
    return torch.where(any_valid, target, ignore)
