"""Strict inference-only runtime for the supervised XR1 residual tail."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import nn


ACTIVE_INDICES = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15)
GROUP_DIMENSIONS = (tuple(range(0, 6)), (6,), tuple(range(7, 13)), (13,))
RESIDUAL_CONTRACT = {
    "version": "robodojo-arx-x5-ee-v2-absolute-gripper-residual-tail-v1",
    "horizon": 10,
    "execution_horizon": 5,
    "full_action_dim": 60,
    "active_dim": 14,
    "active_indices": ACTIVE_INDICES,
    "group_names": ("left_pose", "left_gripper", "right_pose", "right_gripper"),
    "gripper_absolute": True,
}


def canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, block_size: int = 4 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: str | Path) -> str:
    root = Path(path)
    digest = hashlib.sha256()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"checkpoint directory is empty: {root}")
    for item in files:
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def validate_final_acceptance(
    sidecar_path: str | Path,
    *,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    identity: Mapping[str, Any],
    identity_sha256: str,
    tail_runtime_sha256: str,
) -> dict[str, Any]:
    """Verify immutable, independently produced final-test deployment evidence."""
    sidecar_path = Path(sidecar_path)
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"final acceptance sidecar missing: {sidecar_path}")
    value = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if int(value.get("schema_version", -1)) != 1 or value.get("immutable") is not True:
        raise ValueError("final acceptance sidecar schema/immutability marker is invalid")
    recorded_sha256 = value.get("sidecar_sha256")
    payload = {key: item for key, item in value.items() if key != "sidecar_sha256"}
    if recorded_sha256 != canonical_sha256(payload):
        raise ValueError("final acceptance sidecar fingerprint is corrupt")
    if value.get("passed") is not True:
        raise ValueError("final test acceptance did not pass")
    checkpoint = value.get("checkpoint", {})
    if checkpoint.get("file") != Path(checkpoint_path).name:
        raise ValueError("final acceptance names a different checkpoint")
    if checkpoint.get("sha256") != checkpoint_sha256:
        raise ValueError("final acceptance checkpoint SHA256 mismatch")
    evidence = value.get("evidence", {})
    if evidence.get("split") != "test" or evidence.get("acceptance", {}).get("passed") is not True:
        raise ValueError("final acceptance must contain passing independent test evidence")
    hashes = value.get("hashes", {})
    expected_hashes = {
        "run_identity_sha256": identity_sha256,
        "model_config_sha256": identity.get("model_config_sha256"),
        "training_config_sha256": identity.get("training_config_sha256"),
        "cache_manifest_sha256": identity.get("cache_manifest_sha256"),
        "policy_code_sha256": identity.get("policy_code_sha256"),
        "action_contract_sha256": identity.get("action_contract_sha256"),
        "strong_baseline_artifact_sha256": hashes.get("strong_baseline_artifact_sha256"),
        "d0_audit_sha256": identity.get("d0_audit_sha256"),
        "tail_runtime_sha256": tail_runtime_sha256,
    }
    baseline_sha256 = str(hashes.get("strong_baseline_artifact_sha256", ""))
    if len(baseline_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in baseline_sha256):
        raise ValueError("final acceptance lacks a valid strong baseline hash")
    if hashes != expected_hashes:
        raise ValueError("final acceptance provenance hashes differ from run identity")
    return value


def pack_runtime_state(state: Mapping[str, Any]) -> np.ndarray:
    """Match residual_cache_common.pack_state exactly: 7+6+1 per arm."""
    parts = []
    for side in ("left", "right"):
        pose = np.asarray(state[f"{side}_ee_pose"], np.float32).reshape(7)
        joint = np.asarray(state[f"{side}_arm_joint_state"], np.float32).reshape(-1)[:6]
        gripper = np.asarray(state[f"{side}_ee_joint_state"], np.float32).reshape(-1)[:1]
        if joint.shape != (6,) or gripper.shape != (1,):
            raise ValueError(f"invalid {side} state dimensions")
        parts.extend((pose, joint, gripper))
    packed = np.concatenate(parts).astype(np.float32, copy=False)
    if packed.shape != (28,) or not np.isfinite(packed).all():
        raise FloatingPointError("runtime state must be finite [28]")
    return packed


@dataclass(frozen=True)
class ResidualTailConfig:
    horizon: int = 10
    execution_horizon: int = 5
    active_dim: int = 14
    history_length: int = 5
    history_dim: int = 28
    instruction_classes: int = 1
    optional_context_dim: int = 0
    optional_context_tokens: int = 0
    d_model: int = 384
    num_layers: int = 6
    num_heads: int = 6
    dim_feedforward: int = 1536
    dropout: float = 0.1
    stage_classes: int = 7
    contact_dim: int = 2
    logstd_min: float = -5.0
    logstd_max: float = 1.0
    residual_bounds: tuple[float, ...] = (
        0.03, 0.03, 0.03, 0.2617994, 0.2617994, 0.2617994, 0.30,
        0.03, 0.03, 0.03, 0.2617994, 0.2617994, 0.2617994, 0.30,
    )
    right_pose_calibrator_enabled: bool = False
    right_pose_calibrator_width: int = 64
    right_pose_calibrator_gate_mode: str = "open"
    right_pose_calibrator_bounds: tuple[float, ...] = (
        0.01, 0.01, 0.01, 0.05235988, 0.05235988, 0.05235988,
    )
    right_pose_utility_gate_enabled: bool = False
    right_pose_utility_gate_width: int = 64

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ResidualTailConfig":
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown residual-tail config keys: {sorted(unknown)}")
        payload = dict(values)
        payload["residual_bounds"] = tuple(float(value) for value in payload["residual_bounds"])
        if "right_pose_calibrator_bounds" in payload:
            payload["right_pose_calibrator_bounds"] = tuple(
                float(value) for value in payload["right_pose_calibrator_bounds"]
            )
        config = cls(**payload)
        if (config.horizon, config.execution_horizon, config.active_dim) != (10, 5, 14):
            raise ValueError("runtime requires H=10, E=5, active_dim=14")
        if (config.history_length, config.history_dim) != (5, 28):
            raise ValueError("runtime requires training-parity history [5,28]")
        if config.instruction_classes != 1 or config.optional_context_dim != 0:
            raise ValueError("fold v1 runtime requires instruction_classes=1 and no optional context")
        if config.right_pose_calibrator_gate_mode not in {"open", "learned"}:
            raise ValueError("invalid right-pose calibrator gate mode")
        if len(config.right_pose_calibrator_bounds) != 6:
            raise ValueError("right-pose calibrator requires six bounds")
        if config.right_pose_utility_gate_enabled and (
            not config.right_pose_calibrator_enabled
            or config.right_pose_calibrator_width != 128
            or config.right_pose_utility_gate_width != 64
        ):
            raise ValueError("utility gate v1 requires calibrator width 128 and gate width 64")
        return config


class _Projection(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(output_dim, output_dim),
        )

    def forward(self, value):
        return self.network(value)


class ResidualTail(nn.Module):
    """Architecture-identical inference mirror of residual_tail.model.ResidualTail."""
    def __init__(self, config: ResidualTailConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        self.action_projection = _Projection(14, d, config.dropout)
        self.history_projection = _Projection(28, d, config.dropout)
        self.instruction_embedding = nn.Embedding(config.instruction_classes, d)
        self.stage_embedding = nn.Embedding(7, d)
        self.optional_context_projection = None
        self.action_positions = nn.Parameter(torch.zeros(1, 10, d))
        self.history_positions = nn.Parameter(torch.zeros(1, 5, d))
        self.optional_context_positions = None
        self.token_type = nn.Embedding(5, d)
        self.stage_token_bias = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=config.num_heads, dim_feedforward=config.dim_feedforward,
            dropout=config.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, config.num_layers, norm=nn.LayerNorm(d))
        self.mean_head = nn.Linear(d, 14)
        self.logstd_head = nn.Linear(d, 14)
        self.gate_head = nn.Linear(d, 4)
        self.contact_head = nn.Linear(d, 2)
        self.stage_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 7))
        self.register_buffer("residual_bounds", torch.tensor(config.residual_bounds), persistent=True)
        self.right_pose_calibrator = None
        self.right_pose_utility_gate = None
        if config.right_pose_calibrator_enabled:
            width = config.right_pose_calibrator_width
            self.right_pose_calibrator = nn.Sequential(
                nn.LayerNorm(d + 7), nn.Linear(d + 7, width), nn.SiLU(),
                nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 7),
            )
            self.register_buffer(
                "right_pose_calibrator_bounds",
                torch.tensor(config.right_pose_calibrator_bounds),
                persistent=True,
            )
            if config.right_pose_utility_gate_enabled:
                self.right_pose_utility_gate = nn.Sequential(
                    nn.LayerNorm(width + 7), nn.Linear(width + 7, 64), nn.SiLU(),
                    nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1),
                )

    def forward(self, batch):
        base = batch["base_action"]
        history = batch["history"]
        history_valid = batch["history_valid"].bool()
        stage = batch["stage_id"].long()
        instruction = batch["instruction_id"].long()
        valid = batch["valid_mask"].bool()
        stage_context = self.stage_embedding(stage)
        device = base.device
        history_tokens = self.history_projection(history) + self.history_positions + self.token_type(torch.tensor(0, device=device))
        instruction_token = self.instruction_embedding(instruction)[:, None] + self.token_type(torch.tensor(1, device=device))
        stage_token = stage_context[:, None] + self.stage_token_bias + self.token_type(torch.tensor(2, device=device))
        action_tokens = self.action_projection(base) + self.action_positions + stage_context[:, None] + self.token_type(torch.tensor(3, device=device))
        tokens = torch.cat((history_tokens, instruction_token, stage_token, action_tokens), dim=1)
        padding = torch.cat((~history_valid, torch.zeros(len(base), 12, dtype=torch.bool, device=device)), dim=1)
        encoded = self.transformer(tokens, src_key_padding_mask=padding)[:, -10:]
        residual = torch.tanh(self.mean_head(encoded)) * self.residual_bounds
        gate = self.gate_head(encoded).sigmoid()
        calibrator_delta = None
        calibrator_gate = None
        if self.right_pose_calibrator is not None:
            features = torch.cat(
                (encoded, residual[..., 7:13] / self.residual_bounds[7:13], gate[..., 2:3]),
                dim=-1,
            )
            hidden = self.right_pose_calibrator[:-1](features)
            raw = self.right_pose_calibrator[-1](hidden)
            calibrator_delta = torch.tanh(raw[..., :6]) * self.right_pose_calibrator_bounds
            gate_logits = raw[..., 6]
            if self.right_pose_utility_gate is not None:
                utility_input = torch.cat(
                    (
                        hidden,
                        calibrator_delta / self.right_pose_calibrator_bounds,
                        gate[..., 2:3],
                    ),
                    dim=-1,
                )
                gate_logits = self.right_pose_utility_gate(utility_input).squeeze(-1)
            calibrator_gate = (
                torch.ones_like(gate_logits)
                if self.config.right_pose_calibrator_gate_mode == "open"
                else gate_logits.sigmoid()
            )
        return residual, gate, calibrator_delta, calibrator_gate


def _pose_transform(pose: np.ndarray) -> np.ndarray:
    value = np.asarray(pose, np.float64).reshape(7)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(value[[4, 5, 6, 3]]).as_matrix()
    transform[:3, 3] = value[:3]
    return transform


def _se3_exp(twist: np.ndarray) -> np.ndarray:
    velocity, omega = np.asarray(twist, np.float64).reshape(6)[:3], np.asarray(twist, np.float64).reshape(6)[3:]
    theta = np.linalg.norm(omega)
    skew = np.asarray([[0, -omega[2], omega[1]], [omega[2], 0, -omega[0]], [-omega[1], omega[0], 0]])
    if theta < 1e-6:
        jacobian = np.eye(3) + 0.5 * skew + (skew @ skew) / 6.0
    else:
        jacobian = np.eye(3) + (1 - np.cos(theta)) / theta**2 * skew + (theta - np.sin(theta)) / theta**3 * (skew @ skew)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_rotvec(omega).as_matrix()
    result[:3, 3] = jacobian @ velocity
    return result


def compose_action(base_action: Mapping[str, Any], applied: np.ndarray) -> dict[str, np.ndarray]:
    """Compose T_base@Exp(delta) and absolute gripper residuals."""
    residual = np.asarray(applied, np.float32).reshape(14)
    if not np.isfinite(residual).all():
        raise FloatingPointError("residual contains NaN/Inf")
    output = {}
    for side, pose_slice, grip_index in (("left", slice(0, 6), 6), ("right", slice(7, 13), 13)):
        base_pose = np.asarray(base_action[f"{side}_ee_pose"], np.float32).reshape(7)
        composed = _pose_transform(base_pose) @ _se3_exp(residual[pose_slice])
        quat_xyzw = Rotation.from_matrix(composed[:3, :3]).as_quat()
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
        if quat_wxyz[0] < 0:
            quat_wxyz = -quat_wxyz
        output[f"{side}_ee_pose"] = np.concatenate((composed[:3, 3], quat_wxyz)).astype(np.float32)
        base_gripper = float(np.asarray(base_action[f"{side}_ee_joint_state"]).reshape(-1)[0])
        output[f"{side}_ee_joint_state"] = np.asarray([np.clip(base_gripper + residual[grip_index], 0.0, 1.0)], np.float32)
    return output


def compose_right_pose_calibration(
    command: Mapping[str, Any], delta: np.ndarray
) -> dict[str, np.ndarray]:
    output = {key: np.asarray(value).copy() for key, value in command.items()}
    right_pose = np.asarray(command["right_ee_pose"], np.float32).reshape(7)
    composed = _pose_transform(right_pose) @ _se3_exp(np.asarray(delta, np.float32).reshape(6))
    quat_xyzw = Rotation.from_matrix(composed[:3, :3]).as_quat()
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    if quat_wxyz[0] < 0:
        quat_wxyz = -quat_wxyz
    output["right_ee_pose"] = np.concatenate((composed[:3, 3], quat_wxyz)).astype(np.float32)
    return output


class ResidualTailRuntime:
    def __init__(self, model, config, device, identity, checkpoint_sha256, *, evaluation_only=False):
        self.model = model.eval()
        self.config = config
        self.device = torch.device(device)
        self.identity = identity
        self.checkpoint_sha256 = checkpoint_sha256
        self.evaluation_only = bool(evaluation_only)
        self.policy_code_sha256 = identity["policy_code_sha256"]
        self.histories: dict[int, deque[np.ndarray]] = {}

    @classmethod
    def load_task_bank(
        cls, path, *, task_slug, device, expected_base_sha256,
        expected_adapter_sha256, expected_checkpoint_sha256,
        evaluation_only=False,
    ):
        """Load one isolated head from a multi-task bank; disabled routes are identity."""
        path = Path(path)
        actual_sha = sha256_file(path)
        if expected_checkpoint_sha256 is None or actual_sha != expected_checkpoint_sha256:
            raise ValueError("task-bank loading requires an exact SHA256 pin")
        bank = torch.load(path, map_location="cpu", weights_only=False)
        if bank.get("schema_name") != "goai-task-isolated-residual-bank-v1":
            raise ValueError("unsupported task-bank schema")
        raw_task_slug = str(task_slug).strip()
        aliases = bank.get("task_aliases", {})
        if aliases and not isinstance(aliases, dict):
            raise ValueError("task-bank task_aliases must be a mapping")
        task_slug = str(aliases.get(raw_task_slug, raw_task_slug))
        # Backward-compatible handling for older banks without task_aliases.
        if task_slug.endswith("_random"):
            task_slug = task_slug[:-len("_random")]
        if task_slug not in bank.get("task_routes", {}):
            raise KeyError(
                f"task absent from residual bank: raw={raw_task_slug}, canonical={task_slug}"
            )
        route = dict(bank["task_routes"][task_slug])
        route["raw_task_slug"] = raw_task_slug
        route["canonical_task_slug"] = task_slug
        if route.get("base_checkpoint_sha256") != expected_base_sha256:
            raise ValueError("task-bank/base checkpoint mismatch")
        if route.get("semantic_adapter_sha256") != expected_adapter_sha256:
            raise ValueError("task-bank/source adapter mismatch")
        enabled = bool(route.get("deployment_enabled", False))
        if route.get("evaluation_only", False):
            enabled = bool(evaluation_only)
        if not enabled:
            return None, {**route, "fallback": "exact_source_policy"}

        from residual_tail.task_bank_runtime import load_task_model

        model, verified_route = load_task_model(
            path,
            task_slug,
            base_checkpoint_sha256=expected_base_sha256,
            semantic_adapter_sha256=expected_adapter_sha256,
            device=device,
            evaluation_only=evaluation_only,
        )
        if model is None:
            return None, verified_route
        config = model.config
        route = dict(verified_route)
        route["raw_task_slug"] = raw_task_slug
        route["canonical_task_slug"] = task_slug
        identity = {"policy_code_sha256": bank.get("source_policy_code_sha256", "task-bank")}
        return cls(model, config, device, identity, actual_sha,
                   evaluation_only=bool(evaluation_only)), route

    @classmethod
    def load(
        cls, path, *, device, expected_base_sha256, expected_adapter_sha256,
        expected_policy_code_sha256, expected_checkpoint_sha256=None,
        evaluation_only=False,
    ) -> "ResidualTailRuntime":
        path = Path(path)
        sidecar = path.parent / "run_identity.json"
        if not path.is_file() or not sidecar.is_file():
            raise FileNotFoundError(f"residual checkpoint/sidecar missing: {path}, {sidecar}")
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        checkpoint_sha256 = sha256_file(path)
        if expected_checkpoint_sha256 is not None and checkpoint_sha256 != expected_checkpoint_sha256:
            raise ValueError("residual checkpoint SHA256 differs from deployment pin")
        identity = metadata["identity"]
        if metadata.get("identity_sha256") != canonical_sha256(identity):
            raise ValueError("residual sidecar identity hash is corrupt")
        if metadata.get("contract") != json.loads(json.dumps(RESIDUAL_CONTRACT)):
            raise ValueError("residual action contract mismatch")
        if identity.get("action_contract_sha256") != canonical_sha256(RESIDUAL_CONTRACT):
            raise ValueError("residual action-contract fingerprint mismatch")
        config_payload = metadata["model"]
        if identity.get("model_config_sha256") != canonical_sha256(config_payload):
            raise ValueError("residual model config hash mismatch")
        if identity.get("training_config_sha256") != canonical_sha256(metadata["resolved_training_config"]):
            raise ValueError("resolved training config hash mismatch")
        if identity.get("base_checkpoint_sha256") != expected_base_sha256:
            raise ValueError("residual/base checkpoint hash mismatch")
        if identity.get("semantic_adapter_sha256") != expected_adapter_sha256:
            raise ValueError("residual/semantic adapter hash mismatch")
        policy_code_sha256 = str(identity.get("policy_code_sha256", ""))
        if len(policy_code_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in policy_code_sha256):
            raise ValueError("residual identity lacks a valid policy_code_sha256")
        if policy_code_sha256 != expected_policy_code_sha256 and not evaluation_only:
            raise ValueError("residual cache policy code differs from deployed base policy code")
        current_tail_runtime_sha256 = sha256_file(Path(__file__))
        if identity.get("tail_runtime_sha256") != current_tail_runtime_sha256 and not evaluation_only:
            raise ValueError("residual identity tail runtime differs from deployed runtime code")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("schema_version") != 2:
            raise ValueError("unsupported residual checkpoint schema")
        if checkpoint.get("run_identity") != identity or checkpoint.get("run_identity_sha256") != metadata["identity_sha256"]:
            raise ValueError("checkpoint and sidecar identities differ")
        acceptance = checkpoint.get("extra", {}).get("acceptance", {})
        selection = checkpoint.get("extra", {}).get("selection", {})
        if selection.get("intermediate_only") is True:
            raise ValueError(
                "intermediate-only direction checkpoint cannot be loaded by the runtime"
            )
        if evaluation_only:
            if expected_checkpoint_sha256 is None:
                raise ValueError("evaluation-only loading requires an explicit checkpoint SHA256 pin")
            if selection.get("passed") is not True:
                raise ValueError("evaluation-only loading requires selection.passed=true")
            print(
                "[ResidualTailRuntime] EVALUATION ONLY: final test acceptance "
                "is intentionally not treated as deployment authorization"
            )
        else:
            if acceptance.get("passed") is not True:
                raise ValueError("deployment requires a checkpoint with acceptance.passed=true")
            validate_final_acceptance(
                path.parent / "final_acceptance.json",
                checkpoint_path=path,
                checkpoint_sha256=checkpoint_sha256,
                identity=identity,
                identity_sha256=metadata["identity_sha256"],
                tail_runtime_sha256=current_tail_runtime_sha256,
            )
        config = ResidualTailConfig.from_dict(config_payload)
        model = ResidualTail(config)
        shadow = checkpoint.get("ema", {}).get("shadow", {})
        parameters = dict(model.named_parameters())
        if set(shadow) != set(parameters):
            raise ValueError("EMA parameter set differs from runtime architecture")
        with torch.no_grad():
            for name, parameter in parameters.items():
                value = shadow[name]
                if tuple(value.shape) != tuple(parameter.shape) or not torch.isfinite(value).all():
                    raise ValueError(f"invalid EMA tensor {name}")
                parameter.copy_(value.to(parameter))
        model.to(device)
        return cls(
            model, config, device, identity, checkpoint_sha256,
            evaluation_only=evaluation_only,
        )

    def reset(self):
        self.histories.clear()

    def update_history(self, env_key: int, state: Mapping[str, Any]):
        history = self.histories.setdefault(int(env_key), deque(maxlen=5))
        history.append(pack_runtime_state(state))

    @torch.inference_mode()
    def correct(self, *, env_key, obs, base_chunk, base_action, action_index, stage_id, instruction_id=0):
        self.update_history(env_key, obs["state"])
        history_values = list(self.histories[int(env_key)])
        history = np.zeros((5, 28), np.float32)
        valid = np.zeros(5, bool)
        history[-len(history_values):] = history_values
        valid[-len(history_values):] = True
        chunk = np.asarray(base_chunk, np.float32)
        if chunk.shape != (10, 60) or not np.isfinite(chunk).all():
            raise ValueError("stored base chunk must be finite [10,60]")
        batch = {
            "base_action": torch.from_numpy(chunk[:, ACTIVE_INDICES])[None].to(self.device),
            "history": torch.from_numpy(history)[None].to(self.device),
            "history_valid": torch.from_numpy(valid)[None].to(self.device),
            "stage_id": torch.tensor([stage_id], device=self.device),
            "instruction_id": torch.tensor([instruction_id], device=self.device),
            "valid_mask": torch.ones((1, 10), dtype=torch.bool, device=self.device),
        }
        residual, group_gate, calibrator_delta, calibrator_gate = self.model(batch)
        index = int(action_index)
        if not 0 <= index < 5:
            raise ValueError("online action_index must be in execution prefix [0,5)")
        value = residual[0, index]
        gates = group_gate[0, index]
        expanded = torch.empty(14, device=self.device)
        for group, dimensions in enumerate(GROUP_DIMENSIONS):
            expanded[list(dimensions)] = gates[group]
        applied = (value * expanded).float().cpu().numpy()
        bounds = np.asarray(self.config.residual_bounds, np.float32)
        if not np.isfinite(applied).all() or np.any(np.abs(applied) > bounds + 1e-6):
            raise FloatingPointError("tail output violates finite residual bounds")
        command = compose_action(base_action, applied)
        applied_calibration = np.zeros(6, np.float32)
        if calibrator_delta is not None:
            applied_calibration = (
                calibrator_delta[0, index] * calibrator_gate[0, index]
            ).float().cpu().numpy()
            bounds = np.asarray(self.config.right_pose_calibrator_bounds, np.float32)
            if np.any(np.abs(applied_calibration) > bounds + 1e-6):
                raise FloatingPointError("right-pose calibrator violates bounds")
            command = compose_right_pose_calibration(command, applied_calibration)
        return command, {
            "group_gate": gates.float().cpu().numpy(),
            "applied_residual": applied,
            "right_pose_calibration": applied_calibration,
        }
