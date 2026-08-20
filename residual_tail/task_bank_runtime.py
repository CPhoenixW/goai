"""Fail-closed loader for a task-isolated residual-bank checkpoint."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from residual_tail.model import ResidualTail, ResidualTailConfig


def _sha(values: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name].detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(value.dtype).encode() + b"\0")
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_task_model(checkpoint_path: str | Path, task_slug: str, *, base_checkpoint_sha256: str,
                    semantic_adapter_sha256: str | None,
                    device: str | torch.device = "cpu", evaluation_only: bool = False
                    ) -> tuple[ResidualTail | None, dict]:
    bank = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if bank.get("schema_name") != "goai-task-isolated-residual-bank-v1":
        raise ValueError("wrong task-bank schema")
    if task_slug not in bank["task_heads"]:
        raise KeyError(task_slug)
    route = bank["task_routes"][task_slug]
    if route["base_checkpoint_sha256"] != base_checkpoint_sha256:
        raise ValueError("source-policy base checkpoint does not match selected residual head")
    if route["semantic_adapter_sha256"] != semantic_adapter_sha256:
        raise ValueError("source-policy adapter does not match selected residual head")
    enabled = bool(route.get("deployment_enabled", False))
    if route.get("evaluation_only", False):
        enabled = bool(evaluation_only)
    if not enabled:
        fallback = dict(route)
        fallback["fallback"] = "exact_source_policy"
        return None, fallback
    if _sha(bank["shared_state"]) != bank["shared_state_sha256"]:
        raise ValueError("shared-state fingerprint mismatch")
    head = bank["task_heads"][task_slug]
    if _sha(head) != bank["task_head_sha256"][task_slug]:
        raise ValueError("task-head fingerprint mismatch")
    uses_mlp_gate = any(name.startswith("gate_head.0.") for name in head)
    config = ResidualTailConfig(
        history_length=5,
        history_dim=28,
        instruction_classes=1,
        task_gate_mlp_enabled=uses_mlp_gate,
        task_gate_mlp_width=64,
    )
    model = ResidualTail(config)
    model.load_state_dict({**bank["shared_state"], **head}, strict=True)
    model.to(device).eval()
    return model, dict(route)
