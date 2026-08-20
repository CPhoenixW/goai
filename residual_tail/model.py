"""Execution-aware ASRR/ResiP-style residual-tail model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .contract import ACTION_CONTRACT


DEFAULT_RESIDUAL_BOUNDS = (
    0.03,
    0.03,
    0.03,
    0.2617994,
    0.2617994,
    0.2617994,
    0.30,
    0.03,
    0.03,
    0.03,
    0.2617994,
    0.2617994,
    0.2617994,
    0.30,
)

DEFAULT_RIGHT_POSE_CALIBRATOR_BOUNDS = (
    0.01,
    0.01,
    0.01,
    0.05235988,
    0.05235988,
    0.05235988,
)

UTILITY_GATE_INITIAL_PROBABILITY = 0.01


@dataclass(frozen=True)
class ResidualTailConfig:
    horizon: int = 10
    execution_horizon: int = 5
    active_dim: int = 14
    history_length: int = 5
    history_dim: int = 128
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
    residual_bounds: tuple[float, ...] = DEFAULT_RESIDUAL_BOUNDS
    right_pose_calibrator_enabled: bool = False
    right_pose_calibrator_width: int = 64
    right_pose_calibrator_gate_mode: str = "open"
    right_pose_calibrator_bounds: tuple[float, ...] = DEFAULT_RIGHT_POSE_CALIBRATOR_BOUNDS
    right_pose_utility_gate_enabled: bool = False
    right_pose_utility_gate_width: int = 64
    task_gate_mlp_enabled: bool = False
    task_gate_mlp_width: int = 64

    def __post_init__(self) -> None:
        if (
            self.horizon != ACTION_CONTRACT.horizon
            or self.execution_horizon != ACTION_CONTRACT.execution_horizon
            or self.active_dim != ACTION_CONTRACT.active_dim
        ):
            raise ValueError("model config must use H=10, E=5 and active_dim=14")
        if self.history_length <= 0 or self.history_dim <= 0:
            raise ValueError("history dimensions must be positive")
        if self.instruction_classes <= 0:
            raise ValueError("instruction_classes must be positive")
        if (self.optional_context_dim == 0) != (self.optional_context_tokens == 0):
            raise ValueError(
                "optional_context_dim and optional_context_tokens must both be zero or both be positive"
            )
        if self.optional_context_dim < 0 or self.optional_context_tokens < 0:
            raise ValueError("optional context dimensions cannot be negative")
        if self.d_model != 384 or self.num_layers != 6:
            raise ValueError("residual-tail v1 requires a 6-layer d_model=384 Transformer")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if len(self.residual_bounds) != self.active_dim:
            raise ValueError(
                f"expected {self.active_dim} residual bounds, got {len(self.residual_bounds)}"
            )
        if any(value <= 0.0 for value in self.residual_bounds):
            raise ValueError("all residual bounds must be positive")
        if self.stage_classes != 7 or self.contact_dim != 2:
            raise ValueError("fold_clothes v1 requires 7 stages and 2-arm contact labels")
        if self.right_pose_calibrator_width <= 0:
            raise ValueError("right_pose_calibrator_width must be positive")
        if self.right_pose_calibrator_gate_mode not in {"open", "learned"}:
            raise ValueError("right_pose_calibrator_gate_mode must be open or learned")
        if len(self.right_pose_calibrator_bounds) != 6 or any(
            value <= 0.0 for value in self.right_pose_calibrator_bounds
        ):
            raise ValueError("right_pose_calibrator_bounds must contain six positive values")
        if self.right_pose_utility_gate_width <= 0:
            raise ValueError("right_pose_utility_gate_width must be positive")
        if self.task_gate_mlp_width <= 0:
            raise ValueError("task_gate_mlp_width must be positive")
        if self.right_pose_utility_gate_enabled:
            if not self.right_pose_calibrator_enabled:
                raise ValueError("utility gate requires right_pose_calibrator_enabled")
            if self.right_pose_calibrator_width != 128:
                raise ValueError("utility gate v1 requires a 128-wide direction calibrator")
            if self.right_pose_utility_gate_width != 64:
                raise ValueError("utility gate v1 requires width=64")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ResidualTailConfig":
        valid = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - valid)
        if unknown:
            raise ValueError(f"unknown model config keys: {unknown}")
        payload = dict(values)
        if "residual_bounds" in payload:
            payload["residual_bounds"] = tuple(float(v) for v in payload["residual_bounds"])
        if "right_pose_calibrator_bounds" in payload:
            payload["right_pose_calibrator_bounds"] = tuple(
                float(v) for v in payload["right_pose_calibrator_bounds"]
            )
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResidualTailOutput:
    residual_mean: torch.Tensor
    residual_logstd: torch.Tensor
    group_gate_logits: torch.Tensor
    stage_logits: torch.Tensor
    contact_logits: torch.Tensor
    action_features: torch.Tensor
    right_pose_calibrator_delta: torch.Tensor | None = None
    right_pose_calibrator_gate_logits: torch.Tensor | None = None
    right_pose_calibrator_gate: torch.Tensor | None = None
    right_pose_calibrator_bounds: torch.Tensor | None = None
    right_pose_utility_gate_logits: torch.Tensor | None = None

    def to_float32(self) -> "ResidualTailOutput":
        """Promote heads before physical SE(3) loss computation.

        The Transformer may run under BF16 autocast, but geometry and
        uncertainty losses need FP32 range and small-angle precision.
        Casting here preserves gradients back to the BF16 forward graph.
        """

        return ResidualTailOutput(
            residual_mean=self.residual_mean.float(),
            residual_logstd=self.residual_logstd.float(),
            group_gate_logits=self.group_gate_logits.float(),
            stage_logits=self.stage_logits.float(),
            contact_logits=self.contact_logits.float(),
            action_features=self.action_features.float(),
            right_pose_calibrator_delta=(
                None
                if self.right_pose_calibrator_delta is None
                else self.right_pose_calibrator_delta.float()
            ),
            right_pose_calibrator_gate_logits=(
                None
                if self.right_pose_calibrator_gate_logits is None
                else self.right_pose_calibrator_gate_logits.float()
            ),
            right_pose_calibrator_gate=(
                None
                if self.right_pose_calibrator_gate is None
                else self.right_pose_calibrator_gate.float()
            ),
            right_pose_calibrator_bounds=(
                None
                if self.right_pose_calibrator_bounds is None
                else self.right_pose_calibrator_bounds.float()
            ),
            right_pose_utility_gate_logits=(
                None
                if self.right_pose_utility_gate_logits is None
                else self.right_pose_utility_gate_logits.float()
            ),
        )

    @property
    def group_gate(self) -> torch.Tensor:
        return self.group_gate_logits.sigmoid()


class _Projection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class ResidualTail(nn.Module):
    """Refine a frozen H=10 proposal while executing one correction at a time.

    Future *base actions* may attend each other because the full proposal is
    known at the anchor.  No future observation enters the model: observation
    history is causal and the tail is re-queried at every control step.
    """

    def __init__(self, config: ResidualTailConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model

        self.action_projection = _Projection(config.active_dim, d_model, config.dropout)
        self.history_projection = _Projection(config.history_dim, d_model, config.dropout)
        self.instruction_embedding = nn.Embedding(config.instruction_classes, d_model)
        self.stage_embedding = nn.Embedding(config.stage_classes, d_model)
        self.optional_context_projection = (
            _Projection(config.optional_context_dim, d_model, config.dropout)
            if config.optional_context_dim > 0
            else None
        )

        self.action_positions = nn.Parameter(torch.zeros(1, config.horizon, d_model))
        self.history_positions = nn.Parameter(
            torch.zeros(1, config.history_length, d_model)
        )
        self.optional_context_positions = (
            nn.Parameter(torch.zeros(1, config.optional_context_tokens, d_model))
            if config.optional_context_tokens > 0
            else None
        )
        self.token_type = nn.Embedding(5, d_model)
        self.stage_token_bias = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers, norm=nn.LayerNorm(d_model)
        )

        self.mean_head = nn.Linear(d_model, config.active_dim)
        self.logstd_head = nn.Linear(d_model, config.active_dim)
        self.gate_head = (
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, config.task_gate_mlp_width),
                nn.SiLU(),
                nn.Linear(config.task_gate_mlp_width, len(ACTION_CONTRACT.group_names)),
            )
            if config.task_gate_mlp_enabled
            else nn.Linear(d_model, len(ACTION_CONTRACT.group_names))
        )
        self.contact_head = nn.Linear(d_model, config.contact_dim)
        self.stage_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, config.stage_classes)
        )
        self.right_pose_calibrator = None
        self.right_pose_utility_gate = None
        if config.right_pose_calibrator_enabled:
            calibrator_input_dim = d_model + 6 + 1
            width = config.right_pose_calibrator_width
            self.right_pose_calibrator = nn.Sequential(
                nn.LayerNorm(calibrator_input_dim),
                nn.Linear(calibrator_input_dim, width),
                nn.SiLU(),
                nn.Linear(width, width),
                nn.SiLU(),
                nn.Linear(width, 7),
            )
            self.register_buffer(
                "right_pose_calibrator_bounds",
                torch.tensor(config.right_pose_calibrator_bounds, dtype=torch.float32),
                persistent=True,
            )
            if config.right_pose_utility_gate_enabled:
                # 128-D frozen direction feature + 6-D bounded direction +
                # the frozen right-pose group gate = 135 dimensions.
                utility_input_dim = width + 6 + 1
                utility_width = config.right_pose_utility_gate_width
                self.right_pose_utility_gate = nn.Sequential(
                    nn.LayerNorm(utility_input_dim),
                    nn.Linear(utility_input_dim, utility_width),
                    nn.SiLU(),
                    nn.Linear(utility_width, utility_width),
                    nn.SiLU(),
                    nn.Linear(utility_width, 1),
                )

        self.register_buffer(
            "residual_bounds",
            torch.tensor(config.residual_bounds, dtype=torch.float32),
            persistent=True,
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.action_positions, std=0.02)
        nn.init.normal_(self.history_positions, std=0.02)
        if self.optional_context_positions is not None:
            nn.init.normal_(self.optional_context_positions, std=0.02)
        nn.init.normal_(self.stage_token_bias, std=0.02)

        # Exact identity start: residual is zero regardless of the initial gate.
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.logstd_head.weight)
        nn.init.constant_(self.logstd_head.bias, -2.0)
        gate_output = self.gate_head[-1] if isinstance(self.gate_head, nn.Sequential) else self.gate_head
        nn.init.zeros_(gate_output.weight)
        nn.init.constant_(gate_output.bias, -2.0)
        nn.init.zeros_(self.contact_head.bias)
        if self.right_pose_calibrator is not None:
            final = self.right_pose_calibrator[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        if self.right_pose_utility_gate is not None:
            final = self.right_pose_utility_gate[-1]
            nn.init.zeros_(final.weight)
            nn.init.constant_(
                final.bias,
                torch.logit(torch.tensor(UTILITY_GATE_INITIAL_PROBABILITY)).item(),
            )

    def forward(self, batch: Mapping[str, torch.Tensor]) -> ResidualTailOutput:
        base_action = self._required(batch, "base_action")
        history = self._required(batch, "history")
        history_valid = self._required(batch, "history_valid").bool()
        stage_id = self._required(batch, "stage_id").long()
        instruction_id = self._required(batch, "instruction_id").long()
        valid_mask = self._required(batch, "valid_mask").bool()
        optional_context = None
        if self.config.optional_context_dim > 0:
            optional_context = self._required(batch, "optional_context")
        elif "optional_context" in batch:
            raise ValueError(
                "batch supplies optional_context but model optional_context_dim=0; "
                "enable the interface explicitly instead of using ignored/zero placeholders"
            )
        self._validate_shapes(
            base_action,
            history,
            history_valid,
            stage_id,
            instruction_id,
            valid_mask,
            optional_context,
        )

        batch_size = base_action.shape[0]
        device = base_action.device
        stage_context = self.stage_embedding(stage_id)

        history_tokens = (
            self.history_projection(history)
            + self.history_positions
            + self.token_type(torch.tensor(0, device=device))
        )
        instruction_token = (
            self.instruction_embedding(instruction_id)[:, None]
            + self.token_type(torch.tensor(1, device=device))
        )
        stage_token = (
            stage_context[:, None]
            + self.stage_token_bias
            + self.token_type(torch.tensor(2, device=device))
        )
        action_tokens = (
            self.action_projection(base_action)
            + self.action_positions
            + stage_context[:, None]
            + self.token_type(torch.tensor(3, device=device))
        )
        prefix_tokens = [history_tokens, instruction_token, stage_token]
        if optional_context is not None:
            prefix_tokens.append(
                self.optional_context_projection(optional_context)
                + self.optional_context_positions
                + self.token_type(torch.tensor(4, device=device))
            )
        tokens = torch.cat((*prefix_tokens, action_tokens), dim=1)
        context_length = tokens.shape[1] - self.config.horizon
        fixed_prefix = context_length - self.config.history_length
        context_padding = torch.cat(
            (
                ~history_valid,
                torch.zeros(batch_size, fixed_prefix, dtype=torch.bool, device=device),
            ),
            dim=1,
        )
        # The full H10 base proposal is known at the anchor and always remains
        # visible. valid_mask controls supervision only, never input attention.
        action_padding = torch.zeros(batch_size, self.config.horizon, dtype=torch.bool, device=device)
        padding_mask = torch.cat((context_padding, action_padding), dim=1)
        encoded = self.transformer(tokens, src_key_padding_mask=padding_mask)
        action_encoded = encoded[:, -self.config.horizon :]

        residual_mean = torch.tanh(self.mean_head(action_encoded)) * self.residual_bounds
        residual_logstd = self.logstd_head(action_encoded).clamp(
            self.config.logstd_min, self.config.logstd_max
        )
        group_gate_logits = self.gate_head(action_encoded)
        contact_logits = self.contact_head(action_encoded)
        calibrator_delta = None
        calibrator_gate_logits = None
        calibrator_gate = None
        utility_gate_logits = None
        if self.right_pose_calibrator is not None:
            right_bounds = self.residual_bounds[7:13].to(residual_mean)
            calibrator_input = torch.cat(
                (
                    action_encoded,
                    residual_mean[..., 7:13] / right_bounds,
                    group_gate_logits[..., 2:3].sigmoid(),
                ),
                dim=-1,
            )
            calibrator_hidden = self.right_pose_calibrator[:-1](calibrator_input)
            raw_calibration = self.right_pose_calibrator[-1](calibrator_hidden)
            calibrator_delta = (
                torch.tanh(raw_calibration[..., :6])
                * self.right_pose_calibrator_bounds.to(raw_calibration)
            )
            calibrator_gate_logits = raw_calibration[..., 6]
            if self.right_pose_utility_gate is not None:
                utility_input = torch.cat(
                    (
                        calibrator_hidden,
                        calibrator_delta / self.right_pose_calibrator_bounds.to(calibrator_delta),
                        group_gate_logits[..., 2:3].sigmoid(),
                    ),
                    dim=-1,
                )
                utility_gate_logits = self.right_pose_utility_gate(utility_input).squeeze(-1)
                calibrator_gate_logits = utility_gate_logits
            calibrator_gate = (
                torch.ones_like(calibrator_gate_logits)
                if self.config.right_pose_calibrator_gate_mode == "open"
                else calibrator_gate_logits.sigmoid()
            )

        weights = valid_mask.to(action_encoded.dtype)
        pooled = (action_encoded * weights[..., None]).sum(dim=1) / weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        stage_logits = self.stage_head(pooled)

        return ResidualTailOutput(
            residual_mean=residual_mean,
            residual_logstd=residual_logstd,
            group_gate_logits=group_gate_logits,
            stage_logits=stage_logits,
            contact_logits=contact_logits,
            action_features=action_encoded,
            right_pose_calibrator_delta=calibrator_delta,
            right_pose_calibrator_gate_logits=calibrator_gate_logits,
            right_pose_calibrator_gate=calibrator_gate,
            right_pose_calibrator_bounds=(
                None
                if self.right_pose_calibrator is None
                else self.right_pose_calibrator_bounds
            ),
            right_pose_utility_gate_logits=utility_gate_logits,
        )

    @staticmethod
    def _required(batch: Mapping[str, torch.Tensor], key: str) -> torch.Tensor:
        if key not in batch:
            raise KeyError(f"residual-tail batch is missing {key!r}")
        value = batch[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"batch[{key!r}] must be a Tensor")
        if not torch.isfinite(value).all() and value.is_floating_point():
            raise FloatingPointError(f"batch[{key!r}] contains NaN or Inf")
        return value

    def _validate_shapes(
        self,
        base_action: torch.Tensor,
        history: torch.Tensor,
        history_valid: torch.Tensor,
        stage_id: torch.Tensor,
        instruction_id: torch.Tensor,
        valid_mask: torch.Tensor,
        optional_context: torch.Tensor | None,
    ) -> None:
        config = self.config
        batch_size = base_action.shape[0]
        expected = {
            "base_action": (batch_size, config.horizon, config.active_dim),
            "history": (batch_size, config.history_length, config.history_dim),
            "history_valid": (batch_size, config.history_length),
            "stage_id": (batch_size,),
            "instruction_id": (batch_size,),
            "valid_mask": (batch_size, config.horizon),
        }
        actual = {
            "base_action": tuple(base_action.shape),
            "history": tuple(history.shape),
            "history_valid": tuple(history_valid.shape),
            "stage_id": tuple(stage_id.shape),
            "instruction_id": tuple(instruction_id.shape),
            "valid_mask": tuple(valid_mask.shape),
        }
        if config.optional_context_dim > 0:
            expected["optional_context"] = (
                batch_size,
                config.optional_context_tokens,
                config.optional_context_dim,
            )
            actual["optional_context"] = tuple(optional_context.shape)
        errors = [
            f"{key}: expected {shape}, got {actual[key]}"
            for key, shape in expected.items()
            if actual[key] != shape
        ]
        if errors:
            raise ValueError("invalid residual-tail batch shapes: " + "; ".join(errors))
        if torch.any(stage_id < 0) or torch.any(stage_id >= config.stage_classes):
            raise ValueError("stage_id is outside [0, stage_classes)")
        if torch.any(instruction_id < 0) or torch.any(
            instruction_id >= config.instruction_classes
        ):
            raise ValueError("instruction_id is outside [0, instruction_classes)")
        if not torch.all(valid_mask[:, 0]):
            raise ValueError("every sample must have a valid first action")

    @torch.no_grad()
    def sample(
        self,
        output: ResidualTailOutput,
        *,
        deterministic: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if deterministic:
            return output.residual_mean
        noise = torch.randn(
            output.residual_mean.shape,
            dtype=output.residual_mean.dtype,
            device=output.residual_mean.device,
            generator=generator,
        )
        sampled = output.residual_mean + noise * output.residual_logstd.exp()
        bounds = self.residual_bounds.to(sampled)
        return torch.maximum(torch.minimum(sampled, bounds), -bounds)
