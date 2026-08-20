"""Strict RoboDojo ARX-X5 EE action contract used by deployment and tests."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

ACTION_CONTRACT_VERSION = "robodojo-arx-x5-ee-v2-absolute-gripper"
ACTION_HORIZON = 10
ACTION_WIDTH = 60
REPLAN_INTERVAL = 5
ACTIVE_SLOTS = tuple(range(0, 6)) + (7,) + tuple(range(8, 14)) + (15,)
GRIPPER_SLOTS = (7, 15)


def active_action_mask() -> np.ndarray:
    """Return the exact 14-slot mask; all padding including 16:60 is inactive."""
    mask = np.zeros((ACTION_HORIZON, ACTION_WIDTH), dtype=np.int32)
    mask[:, ACTIVE_SLOTS] = 1
    return mask


def validate_raw_action_chunk(raw_actions) -> np.ndarray:
    """Reject legacy 30-step or compact/non-60D action tensors."""
    chunk = np.asarray(raw_actions, dtype=np.float32)
    expected = (ACTION_HORIZON, ACTION_WIDTH)
    if chunk.shape != expected:
        raise ValueError(f"expected raw action shape {expected}, got {chunk.shape}")
    if not np.isfinite(chunk).all():
        raise FloatingPointError("raw action contains NaN or Inf")
    return chunk


def validate_action_mask(mask) -> np.ndarray:
    value = np.asarray(mask, dtype=np.int32)
    expected = active_action_mask()
    if value.shape != expected.shape or not np.array_equal(value, expected):
        raise ValueError("action mask does not match the strict 10x60/14-active contract")
    return value


def identity_tail_step(base_action: Mapping[str, object]) -> dict[str, np.ndarray]:
    """A copy-on-write identity tail; validates finite actuator commands."""
    if not isinstance(base_action, Mapping) or not base_action:
        raise TypeError("base_action must be a non-empty mapping")
    output = {}
    for key, value in base_action.items():
        array = np.asarray(value, dtype=np.float32)
        if not np.isfinite(array).all():
            raise FloatingPointError(f"base action {key!r} contains NaN or Inf")
        output[str(key)] = array.copy()
    for key in ("left_ee_joint_state", "right_ee_joint_state"):
        if key in output and np.any((output[key] < 0.0) | (output[key] > 1.0)):
            raise ValueError(f"absolute gripper command {key!r} must be in [0, 1]")
    return output


def execution_prefix(actions) -> list:
    """Select exactly the deployable E=5 prefix from an H=10 base chunk."""
    values = list(actions)
    if len(values) != ACTION_HORIZON:
        raise ValueError(f"expected {ACTION_HORIZON} decoded actions, got {len(values)}")
    return values[:REPLAN_INTERVAL]
