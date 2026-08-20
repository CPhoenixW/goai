# Copyright (C) 2026 Xiaomi Corporation.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
# file except in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

"""Xiaomi_Robotics_1 policy for XPolicyLab evaluation.

Slot layout (each arm occupies 8 slots of the 60-dim state/action vector):
  Input state is always joint:
    [0:6] left_arm_joint, [7:8] left_gripper,
    [8:14] right_arm_joint, [15:16] right_gripper; every other slot is zero.
  Output action depends on action_type:
    - joint: [0:6] left_arm_joint, [7:8] left_gripper,
             [8:14] right_arm_joint, [15:16] right_gripper (rest ignored).
    - ee:    [0:3] left_xyz, [3:6] left_axis_angle, [7:8] left_gripper,
             [8:11] right_xyz, [11:14] right_axis_angle, [15:16] right_gripper
             (rest ignored; rotation transformed MiBot -> simulator frame).
"""

from __future__ import annotations

import sys
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info

try:
    from .action_contract import (
        ACTION_CONTRACT_VERSION,
        ACTION_HORIZON,
        ACTION_WIDTH,
        ACTIVE_SLOTS,
        identity_tail_step,
        validate_raw_action_chunk,
    )
except ImportError:  # XPolicyLab may load a policy directory as a flat module.
    from action_contract import (
        ACTION_CONTRACT_VERSION,
        ACTION_HORIZON,
        ACTION_WIDTH,
        ACTIVE_SLOTS,
        identity_tail_step,
        validate_raw_action_chunk,
    )

try:
    from .residual_tail_runtime import ResidualTailRuntime, sha256_file, sha256_tree
except ImportError:
    from residual_tail_runtime import ResidualTailRuntime, sha256_file, sha256_tree

# RoboDojo -> MiBot EEF local axis redefinition matrix.
# R_mibot = R_robodojo @ P,  P = Rx(+90°) @ Rz(+90°)
# Only rotation changes; position and gripper are unchanged.
EEF_REFRAME_P = np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=np.float64
)
# Inverse: P^T (orthogonal matrix)
EEF_REFRAME_P_INV = EEF_REFRAME_P.T


def _policy_code_sha256() -> str:
    """Hash the exact policy/action-contract sources that produce bundle actions."""
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ("model.py", "action_contract.py"):
        payload = (root / name).read_bytes()
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()

FOLD_CLOTHES_CANONICAL_PROMPT = "Fold the clothes neatly."
FOLD_CLOTHES_STAGE_PROMPTS = {
    "0": "Reach for and securely grasp the left side of the garment.",
    "1": "Lift and reposition the grasped left side, then release it.",
    "2": "Reach for and securely grasp the right side of the garment.",
    "3": "Lift and reposition the grasped right side, then release it.",
    "4": "Move both grippers to the garment edges and grasp them for the final fold.",
    "5": "Fold the garment over, align the edges, and release both sides.",
    "6": "Retract both arms without disturbing the aligned garment.",
}


class FoldClothesStageFSM:
    """Causal, monotonic phase estimator using only observed gripper state."""

    def __init__(self, stable_observations: int = 1, closed: float = 0.15, opened: float = 0.85):
        self.stable_observations = int(stable_observations)
        if self.stable_observations <= 0:
            raise ValueError("stable_observations must be positive")
        self.closed = float(closed)
        self.opened = float(opened)
        self.reset()

    def reset(self):
        self.stage = 0
        self._count = 0

    def update(self, left: float, right: float) -> str:
        predicates = (
            left <= self.closed,
            left >= self.opened,
            right <= self.closed,
            right >= self.opened,
            left <= self.closed and right <= self.closed,
            left >= self.opened and right >= self.opened,
        )
        if self.stage < len(predicates) and predicates[self.stage]:
            self._count += 1
            if self._count >= self.stable_observations:
                self.stage += 1
                self._count = 0
        else:
            self._count = 0
        return str(self.stage)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _normalize(x: torch.Tensor, norm_info: dict) -> torch.Tensor:
    mode = norm_info["mode"]
    if mode == "gaussian":
        return (x - norm_info["mean"]) / norm_info["std"]
    elif mode == "quantile":
        q01, q99 = norm_info["q01"], norm_info["q99"]
        denom = q99 - q01
        valid = denom.abs() > 1e-5
        safe_denom = torch.where(valid, denom, torch.ones_like(denom))
        result = 2 * (x - q01) / safe_denom - 1
        return torch.where(valid, result, x)
    return x


def _denormalize(x: torch.Tensor, norm_info: dict) -> torch.Tensor:
    mode = norm_info["mode"]
    if mode == "gaussian":
        return x * norm_info["std"] + norm_info["mean"]
    elif mode == "quantile":
        q01, q99 = norm_info["q01"], norm_info["q99"]
        denom = q99 - q01
        valid = denom.abs() > 1e-5
        safe_denom = torch.where(valid, denom, torch.ones_like(denom))
        result = (x + 1) / 2 * safe_denom + q01
        return torch.where(valid, result, x)
    return x


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------


def _center_crop_pil(img: Image.Image, crop_ratio: float) -> Image.Image:
    w, h = 320, 256
    new_w, new_h = int(w * crop_ratio), int(h * crop_ratio)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return (
        img.resize((w, h), Image.BILINEAR)
        .crop((left, top, left + new_w, top + new_h))
        .resize((w, h), Image.BILINEAR)
    )


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------


def _ensure_hwc_uint8(image: Any) -> np.ndarray:
    """Convert observation image to HWC uint8 RGB ndarray."""
    if isinstance(image, (bytes, bytearray, memoryview)):
        import cv2
        buf = np.frombuffer(bytes(image), dtype=np.uint8)
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    image = np.asarray(image)
    if image.ndim == 1 and image.dtype == np.uint8:
        import cv2
        decoded = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    if image.ndim == 3:
        if np.issubdtype(image.dtype, np.floating):
            image = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        return image
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _extract_image(obs: dict, cam_keys: list[str]) -> np.ndarray:
    """Extract image from XPolicyLab observation dict."""
    vision = obs.get("vision", {})
    for key in cam_keys:
        if key not in vision:
            continue
        cam = vision[key]
        if isinstance(cam, dict):
            for img_key in ("color", "colors", "rgb"):
                if img_key in cam:
                    return _ensure_hwc_uint8(cam[img_key])
        else:
            return _ensure_hwc_uint8(cam)
    raise KeyError(f"No image found for camera keys: {cam_keys}")


def _ee_pose_sim_to_mibot(xyz_sim: np.ndarray, quat_wxyz_sim: np.ndarray):
    """Convert an ee pose from simulator frame to MiBot frame.

    Only the EEF local axes are redefined (R_mibot = R_sim @ P); the base-frame
    position is unchanged. Returns (pos, rotm) as float64 for downstream math.
    """
    pos_m = np.asarray(xyz_sim, dtype=np.float64).reshape(3)
    q = np.asarray(quat_wxyz_sim, dtype=np.float64).reshape(4)
    rotm_sim = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    rotm_m = rotm_sim @ EEF_REFRAME_P
    return pos_m, rotm_m


def _ee_pose_mibot_to_sim(pos_mibot: np.ndarray, rotm_mibot: np.ndarray):
    """Convert an ee pose from MiBot frame back to simulator frame.

    Inverse of :func:`_ee_pose_sim_to_mibot`: position unchanged, rotation
    mapped by R_sim = R_mibot @ P^T. Returns (xyz, quat_wxyz) as float32.
    """
    xyz = np.asarray(pos_mibot, dtype=np.float32).reshape(3)  # position unchanged
    rotm_sim = np.asarray(rotm_mibot, dtype=np.float64) @ EEF_REFRAME_P_INV
    quat_xyzw = Rotation.from_matrix(rotm_sim).as_quat()
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]].astype(np.float64)
    # Canonicalize: w >= 0
    if quat_wxyz[0] < 0:
        quat_wxyz = -quat_wxyz
    return xyz, quat_wxyz.astype(np.float32)


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.model_cfg = model_cfg
        self.action_type = model_cfg.get("action_type", "joint")
        if self.action_type not in ("joint", "ee"):
            raise ValueError(
                f"[Xiaomi_Robotics_1] Unsupported action_type: {self.action_type!r}. "
                "Supported values are 'joint' and 'ee'."
            )
        self.env_cfg_type = model_cfg["env_cfg_type"]
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Config
        self.task_id = model_cfg.get("task_id", "robodojo")
        self.default_prompt = model_cfg.get(
            "default_prompt", model_cfg.get("task_name", "Perform the task.")
        )
        self.semantic_stage_control = bool(model_cfg.get("semantic_stage_control", False))
        self.stage_stable_observations = int(model_cfg.get("stage_stable_observations", 1))
        if self.semantic_stage_control and model_cfg.get("task_name") != "fold_clothes":
            raise ValueError("semantic_stage_control currently supports only task_name=fold_clothes")
        self.action_max_length = model_cfg.get("action_max_length", 60)
        self.action_length = int(model_cfg.get("action_length", ACTION_HORIZON))
        self.state_token_length = model_cfg.get("state_token_length", 1)
        self.input_length = int(model_cfg.get("input_length", ACTION_WIDTH))
        self.crop_ratio = model_cfg.get("crop_ratio", 0.95)
        self.policy_code_sha256 = _policy_code_sha256()
        if self.action_length != ACTION_HORIZON or self.input_length != ACTION_WIDTH:
            raise ValueError(
                "RoboDojo ARX-X5 requires exact action shape "
                f"{(ACTION_HORIZON, ACTION_WIDTH)}, got "
                f"{(self.action_length, self.input_length)}"
            )
        self.residual_tail_enabled = False
        self.residual_tail = None
        self.residual_tail_runtime = None
        self._residual_tail_failures = 0

        from src.server.deploy import helper

        # Resolve model_dir: explicit path > checkpoints/<ckpt_name>/
        import os
        model_dir = model_cfg.get("model_dir")
        if not model_dir:
            policy_dir = os.path.dirname(os.path.abspath(__file__))
            ckpt_name = model_cfg.get("ckpt_name")
            if ckpt_name:
                model_dir = os.path.join(policy_dir, "checkpoints", ckpt_name)
            if not model_dir or not os.path.isdir(model_dir):
                raise ValueError(
                    f"[Xiaomi_Robotics_1] model_dir is not set and fallback "
                    f"checkpoints/{ckpt_name}/ does not exist at {model_dir}"
                )

        # Load model
        class _ModelArgs:
            model = model_dir

        print(f"[Xiaomi_Robotics_1] Loading model from {model_dir}...")
        (
            self.model,
            self.action_norms,
            self.state_norms,
            self.action_composition,
            _,
        ) = helper(_ModelArgs())

        # Optional, reversible attention-LoRA. The base checkpoint is loaded
        # first with Xiaomi's unmodified key layout, then the small adapter is
        # injected and loaded strictly.
        adapter_path = os.environ.get("XR1_ADAPTER_PATH")
        if adapter_path:
            if not os.path.isfile(adapter_path):
                raise FileNotFoundError(f"XR1_ADAPTER_PATH does not exist: {adapter_path}")
            from src.models.policy_head.DiT import (
                inject_lora_from_adapter_metadata,
                load_attention_lora_adapter,
            )

            adapter_metadata = inject_lora_from_adapter_metadata(self.model, adapter_path)
            adapter_step = load_attention_lora_adapter(self.model, adapter_path)
            self.model.eval()
            print(
                f"[Xiaomi_Robotics_1] Loaded {adapter_metadata.get('train_mode')} adapter "
                f"step={adapter_step}, targets={len(adapter_metadata.get('targets', []))} "
                f"from {adapter_path}"
            )

        # Build action_dim_mask (same logic as casatwin policy_server.py)
        action_dim = max(
            v[-1] if isinstance(v, (list, tuple)) and isinstance(v[-1], int) else 0
            for v in self.action_composition.values()
            if isinstance(v, (list, tuple))
        )
        self.action_dim_mask = torch.zeros(
            action_dim, dtype=torch.int32, device=self.device
        )
        for component, indexs in self.action_composition.items():
            if not isinstance(indexs, (list, tuple)):
                continue
            if isinstance(indexs[1], (list, tuple)):
                _, (t_start, t_end) = indexs
                self.action_dim_mask[t_start:t_end] = 1
            else:
                start, end = indexs
                if not component.startswith("action_padding"):
                    self.action_dim_mask[start:end] = 1

        # Get action shape from norms
        self.action_shape = None
        for norm in self.action_norms.values():
            if norm["mode"] == "gaussian":
                self.action_shape = norm["mean"].shape
                break
            elif norm["mode"] == "quantile":
                self.action_shape = norm["q01"].shape
                break
        if tuple(self.action_shape or ()) != (ACTION_HORIZON, ACTION_WIDTH):
            raise ValueError(
                f"checkpoint action shape must be {(ACTION_HORIZON, ACTION_WIDTH)}, "
                f"got {self.action_shape}"
            )
        active = tuple(torch.nonzero(self.action_dim_mask, as_tuple=False).flatten().cpu().tolist())
        if active != ACTIVE_SLOTS:
            raise ValueError(f"checkpoint active slots must be {ACTIVE_SLOTS}, got {active}")

        # Load VLM processor (use_fast=True, special tokens include a_i)
        vlm_processor_path = model_cfg.get(
            "vlm_processor_path", "Qwen/Qwen3-VL-4B-Instruct"
        )
        from transformers import AutoProcessor

        special_tokens = {"score": "<score>", "state": "<state>"}
        special_tokens.update({f"a_{i}": f"<a_{i}>" for i in range(self.action_max_length)})
        self.processor = AutoProcessor.from_pretrained(
            vlm_processor_path,
            use_fast=True,
            extra_special_tokens=special_tokens,
        )

        # Internal state
        self._encoded_obs_list: list[dict[str, Any]] = []
        self._stage_fsms: dict[int, FoldClothesStageFSM] = {}
        self._base_action_chunks: dict[int, np.ndarray] = {}

        # Optional supervised residual tail. Absence is an exact identity path;
        # presence is strict and any provenance/config mismatch aborts startup.
        tail_path = os.environ.get("XR1_RESIDUAL_TAIL_PATH")
        task_bank_path = os.environ.get("XR1_RESIDUAL_TASK_BANK_PATH")
        if tail_path and task_bank_path:
            raise ValueError("select either XR1_RESIDUAL_TAIL_PATH or XR1_RESIDUAL_TASK_BANK_PATH")
        tail_disabled = os.environ.get("XR1_RESIDUAL_TAIL_DISABLE", "0").lower() in {
            "1", "true", "yes", "on"
        }
        if task_bank_path and not tail_disabled:
            if self.action_type != "ee":
                raise ValueError("XR1 residual task bank requires action_type=ee")
            base_sha256 = sha256_file(model_dir) if os.path.isfile(model_dir) else sha256_tree(model_dir)
            adapter_sha256 = sha256_file(adapter_path) if adapter_path else None
            raw_task_name = str(model_cfg.get("task_name", "")).strip()
            # RoboDojo launches random-layout evaluations with a distinct
            # ``<task>_random`` task name, while the task-bank head is shared
            # with the corresponding ideal task.
            task_slug = (
                raw_task_name[: -len("_random")]
                if raw_task_name.endswith("_random")
                else raw_task_name
            )
            evaluation_only = os.environ.get(
                "XR1_RESIDUAL_TAIL_EVALUATION_ONLY", "0"
            ).lower() in {"1", "true", "yes", "on"}
            loaded = ResidualTailRuntime.load_task_bank(
                task_bank_path, task_slug=task_slug, device=self.device,
                expected_base_sha256=base_sha256,
                expected_adapter_sha256=adapter_sha256,
                expected_checkpoint_sha256=os.environ.get("XR1_RESIDUAL_TASK_BANK_SHA256"),
                evaluation_only=evaluation_only,
            )
            self.residual_tail_runtime, route = loaded
            self.residual_tail_enabled = self.residual_tail_runtime is not None
            print(f"[Xiaomi_Robotics_1] task-bank route raw_task={raw_task_name} "
                  f"task={task_slug} "
                  f"enabled={self.residual_tail_enabled} route={route}")
        elif tail_path and not tail_disabled:
            if self.action_type != "ee":
                raise ValueError("XR1 residual tail requires action_type=ee")
            tail_evaluation_only = os.environ.get(
                "XR1_RESIDUAL_TAIL_EVALUATION_ONLY", "0"
            ).lower() in {"1", "true", "yes", "on"}
            if tail_evaluation_only and model_cfg.get("task_name") != "fold_clothes":
                raise ValueError(
                    "evaluation-only residual tail is allowlisted only for fold_clothes"
                )
            base_sha256 = sha256_file(model_dir) if os.path.isfile(model_dir) else sha256_tree(model_dir)
            adapter_sha256 = sha256_file(adapter_path) if adapter_path else None
            self.residual_tail_runtime = ResidualTailRuntime.load(
                tail_path,
                device=self.device,
                expected_base_sha256=base_sha256,
                expected_adapter_sha256=adapter_sha256,
                expected_policy_code_sha256=self.policy_code_sha256,
                expected_checkpoint_sha256=os.environ.get("XR1_RESIDUAL_TAIL_SHA256"),
                evaluation_only=tail_evaluation_only,
            )
            self.residual_tail_enabled = True
            print(
                "[Xiaomi_Robotics_1] Loaded residual tail "
                f"evaluation_only={tail_evaluation_only} "
                f"checkpoint_sha256={self.residual_tail_runtime.checkpoint_sha256}"
            )

        print(f"[Xiaomi_Robotics_1] Model loaded. action_shape={self.action_shape}")

    # ------------------------------------------------------------------
    # Observation preprocessing
    # ------------------------------------------------------------------

    def _encode_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Encode a single XPolicyLab obs into intermediate representation.

        Returns a dict with 'messages', 'state_tensor', 'action_condition_length'.
        """
        # Extract images
        head_img = _extract_image(obs, ["cam_head", "cam_high", "head_camera"])
        left_img = _extract_image(
            obs, ["cam_left_wrist", "left_camera", "wrist_left"]
        )
        right_img = _extract_image(
            obs, ["cam_right_wrist", "right_camera", "wrist_right"]
        )


        pil_images = [
            _center_crop_pil(Image.fromarray(head_img), self.crop_ratio),
            _center_crop_pil(Image.fromarray(left_img), self.crop_ratio),
            _center_crop_pil(Image.fromarray(right_img), self.crop_ratio),
        ]

        # Extract state. Input state is always joint, packed into the sparse
        # per-arm 8-slot layout: [0:6] left_arm_joint, [7:8] left_gripper,
        # [8:14] right_arm_joint, [15:16] right_gripper; rest zero.
        state = obs.get("state", {})
        left_arm_joint = np.asarray(
            state["left_arm_joint_state"], dtype=np.float32
        ).reshape(-1)[:6]
        left_gripper = float(
            np.asarray(state["left_ee_joint_state"]).reshape(-1)[0]
        )
        right_arm_joint = np.asarray(
            state["right_arm_joint_state"], dtype=np.float32
        ).reshape(-1)[:6]
        right_gripper = float(
            np.asarray(state["right_ee_joint_state"]).reshape(-1)[0]
        )

        state_padded = np.zeros(self.input_length, dtype=np.float32)
        state_padded[0:6] = left_arm_joint
        state_padded[7] = left_gripper
        state_padded[8:14] = right_arm_joint
        state_padded[15] = right_gripper
        state_tensor = torch.from_numpy(state_padded).bfloat16()  # [60]

        # The model predicts RELATIVE (delta) actions w.r.t. the current state
        # (see mibot GetJointAction / GetEEActionPos / GetEEActionAA, ref_frame="ee").
        # Stash the current absolute state so _actions_to_xpl_format can restore
        # absolute actions. joint: current joints; ee: current ee pose in MiBot frame.
        current_state: dict[str, Any] = {
            "left_arm_joint": left_arm_joint.copy(),
            "right_arm_joint": right_arm_joint.copy(),
        }
        if self.action_type == "ee":
            left_pose = np.asarray(state["left_ee_pose"], dtype=np.float64).reshape(7)
            right_pose = np.asarray(state["right_ee_pose"], dtype=np.float64).reshape(7)
            l_pos_m, l_rotm_m = _ee_pose_sim_to_mibot(left_pose[:3], left_pose[3:7])
            r_pos_m, r_rotm_m = _ee_pose_sim_to_mibot(right_pose[:3], right_pose[3:7])
            current_state.update({
                "left_ee_pos_mibot": l_pos_m,
                "left_ee_rotm_mibot": l_rotm_m,
                "right_ee_pos_mibot": r_pos_m,
                "right_ee_rotm_mibot": r_rotm_m,
            })

        # Build prompt via apply_chat_template
        instruction = self._get_instruction(obs)

        # Base messages: vision + instruction
        base_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "The following observations are captured from multiple views.\n# Ego View\n"},
                    {"type": "image", "image": pil_images[0]},
                    {"type": "text", "text": "\n# Left-Wrist View\n"},
                    {"type": "image", "image": pil_images[1]},
                    {"type": "text", "text": "\n# Right-Wrist View\n"},
                    {"type": "image", "image": pil_images[2]},
                    {"type": "text", "text": f"\nGenerate robot actions for the task:\n{instruction} /no_cot"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "<cot></cot>"}],
            },
        ]

        # Compute action_condition_length from base messages
        base_data = self.processor.apply_chat_template(
            base_messages,
            tokenize=True,
            return_dict=True,
            do_resize=False,
            return_tensors="pt",
        )
        action_condition_length = base_data["input_ids"].size(1)

        # State/action turn
        state_tokens = "".join(
            ["<state>" for _ in range(self.state_token_length)]
        )
        action_tokens = "".join(
            [f"<a_{i}>" for i in range(self.action_length)]
        )
        action_response = f"{action_tokens}<score>"

        action_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Robot state: {state_tokens}"}
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": action_response}],
            },
        ]

        # Full messages (not yet tokenized as batch)
        full_messages = base_messages + action_messages

        return {
            "messages": full_messages,
            "state_tensor": state_tensor,
            "action_condition_length": action_condition_length,
            "current_state": current_state,
            "env_key": int(obs.get("_semantic_env_idx", 0)),
        }

    def _build_batch(
        self, encoded_obs_list: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Build a batched model input from a list of encoded observations."""
        batch_size = len(encoded_obs_list)

        # Tokenize all messages as a batch with padding
        all_messages = [item["messages"] for item in encoded_obs_list]
        batch = self.processor.apply_chat_template(
            all_messages,
            tokenize=True,
            return_dict=True,
            do_resize=False,
            return_tensors="pt",
            padding=True,
        )

        # Stack state tensors [B, 1, 60]
        states = torch.stack(
            [item["state_tensor"] for item in encoded_obs_list], dim=0
        ).unsqueeze(1)  # [B, 1, 60]
        batch["state"] = states

        # action_vlm_condition_segments [B, 2]
        batch["action_vlm_condition_segments"] = torch.tensor(
            [[0, item["action_condition_length"]] for item in encoded_obs_list],
            dtype=torch.int64,
        )

        # Metadata
        batch["task_id"] = self.task_id

        return batch

    def _get_instruction(self, obs: dict[str, Any]) -> str:
        if self.semantic_stage_control:
            stage_id = str(obs.get("_semantic_stage_id", ""))
            if stage_id not in FOLD_CLOTHES_STAGE_PROMPTS:
                raise ValueError(f"missing or invalid causal fold_clothes stage: {stage_id!r}")
            return (
                f"{FOLD_CLOTHES_CANONICAL_PROMPT}\n"
                f"Current objective: {FOLD_CLOTHES_STAGE_PROMPTS[stage_id]}"
            )
        for key in ("instruction", "instructions"):
            if key not in obs:
                continue
            val = obs[key]
            if isinstance(val, list):
                val = val[0] if val else ""
            if isinstance(val, str) and val.strip():
                text = val.strip().rstrip(".") + "."
                return text
        return self.default_prompt

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _run_inference_batch(self, data: dict[str, Any]) -> np.ndarray:
        """Run model inference on a batch, return raw actions [B, T, action_dim]."""
        task_id = data.pop("task_id")
        if task_id not in self.action_norms:
            task_id = list(self.action_norms.keys())[0]

        data.pop("global_rank", None)
        data.pop("rollout_i", None)
        data.pop("step_i", None)

        # Move tensors to device
        model_input = {
            key: (value.to(self.device) if isinstance(value, torch.Tensor) else value)
            for key, value in data.items()
        }

        # State normalization
        if "state" in model_input and task_id in self.state_norms:
            s_norm = self.state_norms[task_id]
            if s_norm["mode"] is not None:
                model_input["state"] = _normalize(model_input["state"], s_norm)

        # Action placeholder + normalization
        a_norm = self.action_norms[task_id]
        batch_size = model_input["input_ids"].shape[0]
        if "action" not in model_input:
            model_input["action"] = torch.zeros(
                (batch_size, *self.action_shape),
                device=self.device,
                dtype=torch.bfloat16,
            )
        elif a_norm["mode"] is not None:
            normed = _normalize(model_input["action"], a_norm)
            if a_norm["mode"] == "gaussian":
                norm_valid = a_norm["std"] > 1e-5
            elif a_norm["mode"] == "quantile":
                norm_valid = (a_norm["q99"] - a_norm["q01"]).abs() > 1e-5
            model_input["action"] = torch.where(
                norm_valid, normed, model_input["action"]
            )

        # Action mask (keep [1, action_dim] for broadcast over [B, T, D])
        if "action_mask" not in model_input:
            model_input["action_mask"] = self.action_dim_mask[None]

        # Inference
        with torch.no_grad():
            action = self.model.generate(model_input)

        # Denormalize
        if a_norm["mode"] is not None:
            denormed = _denormalize(action, a_norm)
            if a_norm["mode"] == "gaussian":
                norm_valid = a_norm["std"] > 1e-5
            elif a_norm["mode"] == "quantile":
                norm_valid = (a_norm["q99"] - a_norm["q01"]).abs() > 1e-5
            action = torch.where(norm_valid, denormed, action)

        # [B, T, dim] -> [B, T, action_dim]
        raw_actions = action.float().cpu().numpy()
        return raw_actions

    # ------------------------------------------------------------------
    # Action postprocessing
    # ------------------------------------------------------------------

    def _actions_to_xpl_format(
        self, raw_actions: np.ndarray, current_state: dict[str, Any]
    ) -> list[dict[str, np.ndarray]]:
        """Convert raw model actions [T, 16] to XPolicyLab action dicts.

        The model predicts RELATIVE (delta) actions w.r.t. the observation's
        current state, matching mibot's training-time transforms. This method
        restores ABSOLUTE actions before returning them; ``current_state`` holds
        the observation's absolute state captured in _encode_observation.

        Behavior depends on self.action_type:
          - joint (GetJointAction): abs_joint = current_joint + delta.
                delta slots: [0:6] left_arm_joint, [8:14] right_arm_joint.
                Grippers ([7:8], [15:16]) are absolute (GetAbsAction).
          - ee (GetEEActionPos/AA, ref_frame="ee"): the delta is expressed in the
                current ee frame (MiBot). Restore in MiBot frame, then map to sim:
                    abs_pos_m  = current_pos_m + current_rotm_m @ delta_pos
                    abs_rotm_m = current_rotm_m @ Rot(delta_axis_angle)
                delta slots: [0:3]/[8:11] xyz, [3:6]/[11:14] axis-angle.
                Grippers ([7:8], [15:16]) are absolute (GetAbsAction).
        """
        raw_actions = self._apply_action_safety(raw_actions)

        action_list = []
        for t in range(raw_actions.shape[0]):
            a = raw_actions[t]

            if self.action_type == "joint":
                left_arm = current_state["left_arm_joint"] + a[0:6]
                right_arm = current_state["right_arm_joint"] + a[8:14]
                action_list.append({
                    "left_arm_joint_state": left_arm.astype(np.float32),
                    "left_ee_joint_state": a[7:8].astype(np.float32),
                    "right_arm_joint_state": right_arm.astype(np.float32),
                    "right_ee_joint_state": a[15:16].astype(np.float32),
                })
            else:
                left_xyz, left_quat = self._restore_abs_ee(
                    a[0:3], a[3:6],
                    current_state["left_ee_pos_mibot"],
                    current_state["left_ee_rotm_mibot"],
                )
                right_xyz, right_quat = self._restore_abs_ee(
                    a[8:11], a[11:14],
                    current_state["right_ee_pos_mibot"],
                    current_state["right_ee_rotm_mibot"],
                )
                action_list.append({
                    "left_ee_pose": np.concatenate([left_xyz, left_quat]).astype(np.float32),
                    "right_ee_pose": np.concatenate([right_xyz, right_quat]).astype(np.float32),
                    "left_ee_joint_state": a[7:8].astype(np.float32),
                    "right_ee_joint_state": a[15:16].astype(np.float32),
                })

        return action_list

    @staticmethod
    def _apply_action_safety(raw_actions: np.ndarray) -> np.ndarray:
        """Return the exact contract-space chunk used by the actuator decoder."""
        safe = validate_raw_action_chunk(raw_actions).copy()
        safe[:, 7] = np.clip(safe[:, 7], 0.0, 1.0)
        safe[:, 15] = np.clip(safe[:, 15], 0.0, 1.0)
        return safe

    def _inference_with_seed(self, encoded_obs: dict[str, Any], seed):
        """Run stochastic DiT inference without perturbing process-global RNG state."""
        if seed is None:
            raw = self._run_inference_batch(self._build_batch([encoded_obs]))[0]
            return raw, None
        seed = int(seed)
        if not 0 <= seed < 2**63:
            raise ValueError("inference_seed must be in [0, 2**63)")
        devices = []
        if self.device.type == "cuda":
            devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=devices, enabled=True):
            # Seed CPU and only the model's CUDA device.  torch.manual_seed()
            # would touch every visible CUDA generator and break isolation.
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(seed)
            raw = self._run_inference_batch(self._build_batch([encoded_obs]))[0]
        return raw, seed

    @staticmethod
    def _restore_abs_ee(
        delta_pos: np.ndarray,
        delta_aa: np.ndarray,
        current_pos_m: np.ndarray,
        current_rotm_m: np.ndarray,
    ):
        """Restore an absolute ee pose (sim frame) from an ee-frame delta.

        Inverse of mibot GetEEActionPos/GetEEActionAA with ref_frame="ee",
        all in the MiBot frame, then converted back to the simulator frame:
            abs_pos_m  = current_pos_m + current_rotm_m @ delta_pos
            abs_rotm_m = current_rotm_m @ Rot(delta_aa)
        Returns (xyz, quat_wxyz) in the simulator frame.
        """
        delta_pos = np.asarray(delta_pos, dtype=np.float64).reshape(3)
        abs_pos_m = current_pos_m + current_rotm_m @ delta_pos
        delta_rotm = Rotation.from_rotvec(
            np.asarray(delta_aa, dtype=np.float64).reshape(3)
        ).as_matrix()
        abs_rotm_m = current_rotm_m @ delta_rotm
        return _ee_pose_mibot_to_sim(abs_pos_m, abs_rotm_m)

    # ------------------------------------------------------------------
    # ModelTemplate interface
    # ------------------------------------------------------------------

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def _apply_semantic_stage(self, obs_list):
        """Update the per-environment semantic FSM without encoding images."""
        staged = []
        for index, obs in enumerate(obs_list):
            env_key = int(obs.get("_semantic_env_idx", index))
            fsm = self._stage_fsms.setdefault(
                env_key, FoldClothesStageFSM(self.stage_stable_observations)
            )
            state = obs.get("state", {})
            left = float(np.asarray(state["left_ee_joint_state"]).reshape(-1)[0])
            right = float(np.asarray(state["right_ee_joint_state"]).reshape(-1)[0])
            item = dict(obs)
            item["_semantic_stage_id"] = fsm.update(left, right)
            staged.append(item)
        return staged

    def update_obs_batch(self, obs_list):
        if self.semantic_stage_control:
            obs_list = self._apply_semantic_stage(obs_list)
        self._encoded_obs_list = [self._encode_observation(obs) for obs in obs_list]

    def update_stage_state(self, obs):
        """Advance the semantic FSM only; keep the action observation intact."""
        stage_ids = self.update_stage_state_batch([obs])
        return stage_ids[0] if stage_ids else None

    def update_stage_state_batch(self, obs_list):
        """Lightweight intra-chunk update: state only, no visual encoding."""
        if not self.semantic_stage_control:
            return []
        staged = self._apply_semantic_stage(obs_list)
        return [item["_semantic_stage_id"] for item in staged]

    def get_action(self, **kwargs):
        if not self._encoded_obs_list:
            raise AssertionError(
                "[Xiaomi_Robotics_1] Call update_obs before get_action."
            )
        return self._predict_action_chunk(self._encoded_obs_list[0])

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if not self._encoded_obs_list:
            raise AssertionError(
                "[Xiaomi_Robotics_1] Call update_obs_batch before get_action_batch."
            )
        return [
            self._predict_action_chunk(encoded_obs)
            for encoded_obs in self._encoded_obs_list
        ]

    def _predict_action_chunk(
        self, encoded_obs: dict[str, Any]
    ) -> list[dict[str, np.ndarray]]:
        """Run inference on a single encoded observation."""
        batch_data = self._build_batch([encoded_obs])
        raw_actions = self._run_inference_batch(batch_data)  # [1, T, 16]
        return self._actions_to_xpl_format(
            raw_actions[0], encoded_obs["current_state"]
        )

    def get_action_bundle(self, request=None):
        """Return raw and decoded actions for an auditable residual cache.

        This uses the exact deployment preprocessing/inference path and does
        not change model state.  It is intentionally a public websocket method
        so the cache builder never reimplements Xiaomi normalization.
        """
        if not self._encoded_obs_list:
            raise AssertionError(
                "[Xiaomi_Robotics_1] Call update_obs before get_action_bundle."
            )
        encoded_obs = self._encoded_obs_list[0]
        request = {} if request is None else request
        if not isinstance(request, dict):
            raise TypeError("get_action_bundle request must be a mapping")
        raw_actions_pre_safety, effective_seed = self._inference_with_seed(
            encoded_obs, request.get("inference_seed")
        )
        raw_actions = self._apply_action_safety(raw_actions_pre_safety)
        self._base_action_chunks[int(encoded_obs["env_key"])] = raw_actions.copy()
        decoded_actions = self._actions_to_xpl_format(
            raw_actions, encoded_obs["current_state"]
        )
        if len(decoded_actions) != ACTION_HORIZON:
            raise ValueError(f"decoded action count must be {ACTION_HORIZON}")
        return {
            "raw_actions": raw_actions,
            "raw_actions_pre_safety": raw_actions_pre_safety,
            "decoded_actions": decoded_actions,
            "effective_inference_seed": effective_seed,
            "policy_code_sha256": self.policy_code_sha256,
            "contract_version": ACTION_CONTRACT_VERSION,
            "action_shape": [ACTION_HORIZON, ACTION_WIDTH],
            "active_slots": list(ACTIVE_SLOTS),
        }

    def get_action_bundle_batch(self, request=None):
        if not self._encoded_obs_list:
            raise AssertionError(
                "[Xiaomi_Robotics_1] Call update_obs_batch before get_action_bundle_batch."
            )
        request = {} if request is None else request
        if not isinstance(request, dict):
            raise TypeError("get_action_bundle_batch request must be a mapping")
        seeds = request.get("inference_seeds")
        if seeds is None:
            seeds = [None] * len(self._encoded_obs_list)
        if len(seeds) != len(self._encoded_obs_list):
            raise ValueError("inference_seeds must match encoded batch length")
        bundles = []
        for encoded_obs, seed in zip(self._encoded_obs_list, seeds):
            raw_actions_pre_safety, effective_seed = self._inference_with_seed(encoded_obs, seed)
            raw_actions = self._apply_action_safety(raw_actions_pre_safety)
            self._base_action_chunks[int(encoded_obs["env_key"])] = raw_actions.copy()
            decoded_actions = self._actions_to_xpl_format(
                raw_actions, encoded_obs["current_state"]
            )
            bundles.append({
                "raw_actions": raw_actions,
                "raw_actions_pre_safety": raw_actions_pre_safety,
                "decoded_actions": decoded_actions,
                "effective_inference_seed": effective_seed,
                "policy_code_sha256": self.policy_code_sha256,
                "contract_version": ACTION_CONTRACT_VERSION,
                "action_shape": [ACTION_HORIZON, ACTION_WIDTH],
                "active_slots": list(ACTIVE_SLOTS),
            })
        return bundles

    def apply_residual_tail(self, base_action, obs=None, action_index=0, bundle_metadata=None):
        """Per-control-step tail hook; default is a strict identity copy.

        A learned tail can be attached to ``self.residual_tail`` without
        changing base chunk generation.  Enabling a missing tail fails closed.
        """
        # Generic websocket calls carry one positional request mapping, while
        # direct Python callers use the ordinary explicit arguments.
        if isinstance(base_action, dict) and "base_action" in base_action:
            request = base_action
            base_action = request["base_action"]
            obs = request.get("obs")
            action_index = request.get("action_index", 0)
            bundle_metadata = request.get("bundle_metadata")
        action_index = int(action_index)
        if not 0 <= action_index < ACTION_HORIZON:
            raise ValueError(f"action_index must be in [0, {ACTION_HORIZON})")
        if not self.residual_tail_enabled:
            return identity_tail_step(base_action)
        if self.residual_tail_runtime is not None:
            obs = {} if obs is None else obs
            env_key = int(obs.get("_semantic_env_idx", 0))
            base_chunk = self._base_action_chunks.get(env_key)
            if base_chunk is None:
                return identity_tail_step(base_action)
            fsm = self._stage_fsms.get(env_key)
            stage_id = int(fsm.stage) if fsm is not None else 0
            try:
                corrected, _diagnostics = self.residual_tail_runtime.correct(
                    env_key=env_key,
                    obs=obs,
                    base_chunk=base_chunk,
                    base_action=identity_tail_step(base_action),
                    action_index=action_index,
                    stage_id=stage_id,
                    instruction_id=0,
                )
                return identity_tail_step(corrected)
            except Exception as error:
                # Runtime corrections are optional: invalid/non-finite inputs or
                # outputs must never reach the robot.
                self._residual_tail_failures += 1
                print(f"[Xiaomi_Robotics_1] residual tail fail-closed: {error}")
                return identity_tail_step(base_action)
        if self.residual_tail is None or not callable(self.residual_tail):
            raise RuntimeError("residual_tail_enabled=True but no callable tail is loaded")
        corrected = self.residual_tail(
            obs=obs,
            base_action=identity_tail_step(base_action),
            action_index=action_index,
            bundle_metadata=bundle_metadata or {},
        )
        return identity_tail_step(corrected)

    def apply_residual_tail_batch(
        self, base_action_list, obs_list=None, action_index=0, bundle_metadata_list=None
    ):
        if isinstance(base_action_list, dict) and "base_action_list" in base_action_list:
            request = base_action_list
            base_action_list = request["base_action_list"]
            obs_list = request.get("obs_list")
            action_index = request.get("action_index", 0)
            bundle_metadata_list = request.get("bundle_metadata_list")
        action_index = int(action_index)
        count = len(base_action_list)
        observations = [None] * count if obs_list is None else list(obs_list)
        metadata = [None] * count if bundle_metadata_list is None else list(bundle_metadata_list)
        if len(observations) != count or len(metadata) != count:
            raise ValueError("tail batch inputs must have equal lengths")
        return [
            self.apply_residual_tail({
                "base_action": action,
                "obs": obs,
                "action_index": action_index,
                "bundle_metadata": item_metadata,
            })
            for action, obs, item_metadata in zip(base_action_list, observations, metadata)
        ]

    def set_residual_tail_enabled(self, enabled=True):
        if isinstance(enabled, dict):
            enabled = enabled.get("enabled", True)
        requested = bool(enabled)
        if requested and self.residual_tail_runtime is None and not callable(self.residual_tail):
            raise RuntimeError("cannot enable an unloaded residual tail")
        self.residual_tail_enabled = requested
        return {"enabled": self.residual_tail_enabled}

    def reset(self):
        self._encoded_obs_list = []
        for fsm in self._stage_fsms.values():
            fsm.reset()
        self._stage_fsms = {}
        self._base_action_chunks = {}
        if self.residual_tail_runtime is not None:
            self.residual_tail_runtime.reset()
        print("[Xiaomi_Robotics_1] Model reset.")
