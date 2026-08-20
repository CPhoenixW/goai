"""Frozen-Xiaomi supervised residual-tail training package."""

from .contract import ACTION_CONTRACT, ActionContract
from .model import ResidualTail, ResidualTailConfig, ResidualTailOutput

__all__ = [
    "ACTION_CONTRACT",
    "ActionContract",
    "ResidualTail",
    "ResidualTailConfig",
    "ResidualTailOutput",
]
