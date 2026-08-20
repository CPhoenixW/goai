"""Physical action contract and differentiable SE(3) utilities.

The frozen Xiaomi model keeps a sparse 60-dimensional action tensor.  The
residual tail never edits that tensor directly: it consumes the 14 active
coordinates and predicts two local SE(3) corrections plus two absolute
gripper corrections.  Pose corrections are composed in physical space.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import torch


def _skew(value: torch.Tensor) -> torch.Tensor:
    if value.shape[-1] != 3:
        raise ValueError(f"skew expects [..., 3], got {tuple(value.shape)}")
    x, y, z = value.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    return torch.stack(
        (zeros, -z, y, z, zeros, -x, -y, x, zeros), dim=-1
    ).reshape(*value.shape[:-1], 3, 3)


def _vee(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"vee expects [..., 3, 3], got {tuple(matrix.shape)}")
    return torch.stack(
        (matrix[..., 2, 1], matrix[..., 0, 2], matrix[..., 1, 0]), dim=-1
    )


def _series_a(theta2: torch.Tensor) -> torch.Tensor:
    return 1.0 - theta2 / 6.0 + theta2.square() / 120.0


def _series_b(theta2: torch.Tensor) -> torch.Tensor:
    return 0.5 - theta2 / 24.0 + theta2.square() / 720.0


def _series_c(theta2: torch.Tensor) -> torch.Tensor:
    return 1.0 / 6.0 - theta2 / 120.0 + theta2.square() / 5040.0


def so3_exp(axis_angle: torch.Tensor) -> torch.Tensor:
    """Exponentiate axis-angle vectors into rotation matrices."""

    if axis_angle.shape[-1] != 3:
        raise ValueError(
            f"so3_exp expects [..., 3], got {tuple(axis_angle.shape)}"
        )
    theta2 = axis_angle.square().sum(dim=-1, keepdim=True)
    # Float32 closed forms suffer cancellation well before theta reaches zero.
    # Use the analytic series below 1e-2 rad (theta^2 < 1e-4).
    small = theta2 < 1e-4
    safe_theta2 = theta2.clamp_min(1e-8)
    # Never differentiate sqrt at exactly zero.  torch.where does not protect
    # the inactive branch from 0 * inf during backward, so computing the raw
    # sqrt before the small-angle branch produces NaN gradients at the
    # zero-initialized residual head.
    safe_theta = safe_theta2.sqrt()
    a = torch.where(
        small, _series_a(theta2), torch.sin(safe_theta) / safe_theta
    )
    b = torch.where(
        small,
        _series_b(theta2),
        (1.0 - torch.cos(safe_theta)) / safe_theta2,
    )
    omega = _skew(axis_angle)
    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    eye = eye.expand(*axis_angle.shape[:-1], 3, 3)
    return eye + a[..., None] * omega + b[..., None] * (omega @ omega)


def so3_log(rotation: torch.Tensor) -> torch.Tensor:
    """Logarithm of an SO(3) matrix with stable small-angle handling."""

    if rotation.shape[-2:] != (3, 3):
        raise ValueError(
            f"so3_log expects [..., 3, 3], got {tuple(rotation.shape)}"
        )
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cosine)
    sine = torch.sin(theta)
    skew_vector = _vee(rotation - rotation.transpose(-1, -2))
    scale = theta / (2.0 * sine.clamp_min(1e-7))
    result = scale[..., None] * skew_vector

    small = theta < 1e-4
    result = torch.where(small[..., None], 0.5 * skew_vector, result)

    # The residual trust region is far below pi, but cached expert/base pairs
    # may contain a large disagreement.  Use the principal eigenvector of
    # (R + I) / 2 for a finite near-pi target instead of dividing by sin(pi).
    near_pi = theta > torch.pi - 1e-3
    if torch.any(near_pi):
        eye = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
        symmetric = 0.5 * (rotation + eye)
        _, eigenvectors = torch.linalg.eigh(symmetric)
        axis = eigenvectors[..., -1]
        sign_probe = (axis * skew_vector).sum(dim=-1, keepdim=True)
        axis = torch.where(sign_probe < 0.0, -axis, axis)
        pi_result = axis * theta[..., None]
        result = torch.where(near_pi[..., None], pi_result, result)
    return result


def se3_exp(twist: torch.Tensor) -> torch.Tensor:
    """Exponentiate local twists ordered as ``[translation, rotation]``."""

    if twist.shape[-1] != 6:
        raise ValueError(f"se3_exp expects [..., 6], got {tuple(twist.shape)}")
    velocity, omega_value = twist[..., :3], twist[..., 3:]
    theta2 = omega_value.square().sum(dim=-1, keepdim=True)
    # In float32, ``1 - cos(theta)`` loses precision below a milliradian.
    # The analytic V series is both more accurate and has finite gradients.
    small = theta2 < 1e-4
    safe_theta2 = theta2.clamp_min(1e-8)
    safe_theta = safe_theta2.sqrt()
    safe_theta3 = safe_theta2 * safe_theta
    b = torch.where(
        small, _series_b(theta2), (1.0 - torch.cos(safe_theta)) / safe_theta2
    )
    c = torch.where(
        small,
        _series_c(theta2),
        (safe_theta - torch.sin(safe_theta)) / safe_theta3,
    )
    omega = _skew(omega_value)
    eye3 = torch.eye(3, dtype=twist.dtype, device=twist.device)
    eye3 = eye3.expand(*twist.shape[:-1], 3, 3)
    jacobian = eye3 + b[..., None] * omega + c[..., None] * (omega @ omega)
    translation = (jacobian @ velocity[..., None]).squeeze(-1)

    transform = torch.zeros(
        *twist.shape[:-1], 4, 4, dtype=twist.dtype, device=twist.device
    )
    transform[..., :3, :3] = so3_exp(omega_value)
    transform[..., :3, 3] = translation
    transform[..., 3, 3] = 1.0
    return transform


def se3_log(transform: torch.Tensor) -> torch.Tensor:
    """Logarithm of homogeneous transforms as local 6D twists."""

    if transform.shape[-2:] != (4, 4):
        raise ValueError(
            f"se3_log expects [..., 4, 4], got {tuple(transform.shape)}"
        )
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    omega_value = so3_log(rotation)
    theta2 = omega_value.square().sum(dim=-1, keepdim=True)
    omega = _skew(omega_value)
    eye3 = torch.eye(3, dtype=transform.dtype, device=transform.device)
    eye3 = eye3.expand(*transform.shape[:-2], 3, 3)
    # The closed-form inverse Jacobian suffers the same float32 cancellation.
    small = theta2 < 1e-4
    safe_theta = theta2.clamp_min(1e-8).sqrt()
    coefficient = (
        1.0
        - 0.5
        * safe_theta
        * torch.sin(safe_theta)
        / (1.0 - torch.cos(safe_theta)).clamp_min(1e-8)
    ) / theta2.clamp_min(1e-8)
    coefficient = torch.where(
        small,
        torch.full_like(coefficient, 1.0 / 12.0),
        coefficient,
    )
    jacobian_inv = eye3 - 0.5 * omega + coefficient[..., None] * (omega @ omega)
    velocity = (jacobian_inv @ translation[..., None]).squeeze(-1)
    return torch.cat((velocity, omega_value), dim=-1)


def invert_transform(transform: torch.Tensor) -> torch.Tensor:
    if transform.shape[-2:] != (4, 4):
        raise ValueError(
            f"invert_transform expects [..., 4, 4], got {tuple(transform.shape)}"
        )
    rotation_t = transform[..., :3, :3].transpose(-1, -2)
    translation = -(rotation_t @ transform[..., :3, 3, None]).squeeze(-1)
    result = torch.zeros_like(transform)
    result[..., :3, :3] = rotation_t
    result[..., :3, 3] = translation
    result[..., 3, 3] = 1.0
    return result


def compose_transform(base: torch.Tensor, local_twist: torch.Tensor) -> torch.Tensor:
    if base.shape[:-2] != local_twist.shape[:-1]:
        raise ValueError(
            "base/local_twist batch dimensions differ: "
            f"{tuple(base.shape)} versus {tuple(local_twist.shape)}"
        )
    return base @ se3_exp(local_twist)


def relative_twist(base: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if base.shape != target.shape:
        raise ValueError(
            f"base and target poses differ: {tuple(base.shape)} vs {tuple(target.shape)}"
        )
    return se3_log(invert_transform(base) @ target)


@dataclass(frozen=True)
class ActionContract:
    """Immutable residual-tail action ABI."""

    version: str = "robodojo-arx-x5-ee-v2-absolute-gripper-residual-tail-v1"
    horizon: int = 10
    execution_horizon: int = 5
    full_action_dim: int = 60
    active_dim: int = 14
    active_indices: tuple[int, ...] = (
        0,
        1,
        2,
        3,
        4,
        5,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        15,
    )
    group_names: tuple[str, ...] = (
        "left_pose",
        "left_gripper",
        "right_pose",
        "right_gripper",
    )
    gripper_absolute: bool = True

    def __post_init__(self) -> None:
        if self.horizon != 10 or self.execution_horizon != 5:
            raise ValueError("residual-tail v1 requires H=10 and E=5")
        if self.active_dim != 14 or len(self.active_indices) != 14:
            raise ValueError("residual-tail v1 requires exactly 14 active coordinates")
        if len(set(self.active_indices)) != len(self.active_indices):
            raise ValueError("active indices must be unique")
        if max(self.active_indices) >= self.full_action_dim:
            raise ValueError("active index exceeds full action dimension")
        if not self.gripper_absolute:
            raise ValueError("residual-tail v1 requires absolute gripper commands")

    @property
    def group_dimension_indices(self) -> tuple[tuple[int, ...], ...]:
        return (
            tuple(range(0, 6)),
            (6,),
            tuple(range(7, 13)),
            (13,),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def validate_metadata(self, metadata: Mapping[str, object]) -> None:
        expected = self.to_dict()
        for key in (
            "version",
            "horizon",
            "execution_horizon",
            "full_action_dim",
            "active_dim",
            "gripper_absolute",
        ):
            if metadata.get(key) != expected[key]:
                raise ValueError(
                    f"action contract mismatch for {key}: "
                    f"expected {expected[key]!r}, got {metadata.get(key)!r}"
                )
        indices = tuple(int(value) for value in metadata.get("active_indices", ()))
        if indices != self.active_indices:
            raise ValueError(
                f"active indices mismatch: {indices} != {self.active_indices}"
            )

    def full_to_active(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-1] != self.full_action_dim:
            raise ValueError(
                f"full action expects dim {self.full_action_dim}, got {action.shape[-1]}"
            )
        index = torch.tensor(
            self.active_indices, dtype=torch.long, device=action.device
        )
        return action.index_select(-1, index)

    def active_to_full(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-1] != self.active_dim:
            raise ValueError(
                f"active action expects dim {self.active_dim}, got {action.shape[-1]}"
            )
        result = torch.zeros(
            *action.shape[:-1],
            self.full_action_dim,
            dtype=action.dtype,
            device=action.device,
        )
        index = torch.tensor(
            self.active_indices, dtype=torch.long, device=action.device
        )
        return result.index_copy(-1, index, action)

    def assert_inactive_zero(
        self, action: torch.Tensor, *, atol: float = 0.0
    ) -> None:
        if action.shape[-1] != self.full_action_dim:
            raise ValueError(
                f"full action expects dim {self.full_action_dim}, got {action.shape[-1]}"
            )
        active = set(self.active_indices)
        inactive = [i for i in range(self.full_action_dim) if i not in active]
        values = action[..., inactive]
        if not torch.all(values.abs() <= atol):
            maximum = float(values.abs().max().detach().cpu())
            raise ValueError(
                f"inactive action slots are non-zero: max={maximum}, atol={atol}"
            )

    def expand_group_values(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != len(self.group_names):
            raise ValueError(
                f"group tensor expects dim {len(self.group_names)}, got {values.shape[-1]}"
            )
        result = torch.empty(
            *values.shape[:-1], self.active_dim, dtype=values.dtype, device=values.device
        )
        for group, dimensions in enumerate(self.group_dimension_indices):
            result[..., list(dimensions)] = values[..., group, None]
        return result

    def split_residual(
        self, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual.shape[-1] != self.active_dim:
            raise ValueError(
                f"residual expects dim {self.active_dim}, got {residual.shape[-1]}"
            )
        pose = torch.stack((residual[..., :6], residual[..., 7:13]), dim=-2)
        gripper = torch.stack((residual[..., 6], residual[..., 13]), dim=-1)
        return pose, gripper

    def join_residual(
        self, pose: torch.Tensor, gripper: torch.Tensor
    ) -> torch.Tensor:
        if pose.shape[-2:] != (2, 6):
            raise ValueError(f"pose residual expects [..., 2, 6], got {tuple(pose.shape)}")
        if gripper.shape != pose.shape[:-2] + (2,):
            raise ValueError(
                f"gripper residual shape {tuple(gripper.shape)} is incompatible with pose {tuple(pose.shape)}"
            )
        return torch.cat(
            (pose[..., 0, :], gripper[..., 0:1], pose[..., 1, :], gripper[..., 1:2]),
            dim=-1,
        )

    def residual_target(
        self,
        base_pose: torch.Tensor,
        expert_pose: torch.Tensor,
        base_gripper: torch.Tensor,
        expert_gripper: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_pose_gripper(base_pose, base_gripper)
        self._validate_pose_gripper(expert_pose, expert_gripper)
        if base_pose.shape != expert_pose.shape:
            raise ValueError("base and expert pose shapes differ")
        if base_gripper.shape != expert_gripper.shape:
            raise ValueError("base and expert gripper shapes differ")
        pose_target = relative_twist(base_pose, expert_pose)
        gripper_target = expert_gripper - base_gripper
        return self.join_residual(pose_target, gripper_target)

    def compose(
        self,
        base_pose: torch.Tensor,
        base_gripper: torch.Tensor,
        residual: torch.Tensor,
        *,
        group_gate: torch.Tensor | None = None,
        alpha: float = 1.0,
        residual_bounds: torch.Tensor | Sequence[float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_pose_gripper(base_pose, base_gripper)
        if residual.shape != base_pose.shape[:-3] + (self.active_dim,):
            raise ValueError(
                f"residual shape {tuple(residual.shape)} incompatible with pose {tuple(base_pose.shape)}"
            )
        applied = residual
        if residual_bounds is not None:
            bounds = torch.as_tensor(
                residual_bounds, dtype=residual.dtype, device=residual.device
            )
            if bounds.shape != (self.active_dim,) or torch.any(bounds <= 0.0):
                raise ValueError(
                    f"residual bounds must be positive [{self.active_dim}], got {tuple(bounds.shape)}"
                )
            applied = torch.maximum(torch.minimum(applied, bounds), -bounds)
        if group_gate is not None:
            applied = applied * self.expand_group_values(group_gate)
        applied = applied * float(alpha)
        pose_residual, gripper_residual = self.split_residual(applied)
        composed_pose = compose_transform(base_pose, pose_residual)
        composed_gripper = (base_gripper + gripper_residual).clamp(0.0, 1.0)
        return composed_pose, composed_gripper, applied

    @staticmethod
    def _validate_pose_gripper(
        pose: torch.Tensor, gripper: torch.Tensor
    ) -> None:
        if pose.shape[-3:] != (2, 4, 4):
            raise ValueError(f"pose expects [..., 2, 4, 4], got {tuple(pose.shape)}")
        if gripper.shape != pose.shape[:-3] + (2,):
            raise ValueError(
                f"gripper shape {tuple(gripper.shape)} incompatible with pose {tuple(pose.shape)}"
            )
        if not torch.isfinite(pose).all() or not torch.isfinite(gripper).all():
            raise FloatingPointError("pose/gripper contains NaN or Inf")


ACTION_CONTRACT = ActionContract()
