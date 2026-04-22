# SPDX-License-Identifier: GPL-3.0-or-later
from .base import BaseGradientOP
from .finite_difference import FiniteDifferenceOP
from .finite_volume import FiniteVolumeOP


def create_gradient_op(type: str, **kwargs) -> BaseGradientOP:
    """Factory function to create a gradient operator instance based on the specified type."""
    op_type = type.lower()
    if op_type == "finite_volume":
        return FiniteVolumeOP(**kwargs)
    elif op_type == "finite_difference":
        return FiniteDifferenceOP(**kwargs)
    else:
        raise ValueError(f"Unsupported gradient operator type: {op_type}")


__all__ = [
    "BaseGradientOP",
    "FiniteVolumeOP",
    "FiniteDifferenceOP",
    "create_gradient_op",
]
