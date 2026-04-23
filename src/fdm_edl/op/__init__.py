# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from .axisymmetric.fd import AxisymmetricFDOp
from .axisymmetric.fv import AxisymmetricFVOp
from .base import BaseGradientOP
from .cartesian.fd import EuclideanFDOp
from .cartesian.fv import EuclideanFVOp

# (discretization, coordinate_system) -> operator class
_GRADIENT_OPS = {
    ("finite_volume", "cartesian"): EuclideanFVOp,
    ("finite_volume", "axisymmetric"): AxisymmetricFVOp,
    ("finite_difference", "cartesian"): EuclideanFDOp,
    ("finite_difference", "axisymmetric"): AxisymmetricFDOp,
}


def create_gradient_op(type: str, **kwargs) -> BaseGradientOP:
    """Factory function to create a gradient operator instance based on the specified type."""
    op_type = type.lower()
    coord = kwargs.pop("coordinate_system", "cartesian").lower()

    try:
        cls = _GRADIENT_OPS[(op_type, coord)]
    except KeyError as e:
        supported_types = sorted({t for (t, _) in _GRADIENT_OPS})
        supported_coords = sorted({c for (_, c) in _GRADIENT_OPS})
        raise ValueError(
            f"Unsupported gradient operator: type={op_type!r}, coordinate_system={coord!r}. "
            f"Supported types={supported_types}, coordinate_systems={supported_coords}."
        ) from e

    return cls(**kwargs)


__all__ = [
    "BaseGradientOP",
    "EuclideanFVOp",
    "EuclideanFDOp",
    "AxisymmetricFVOp",
    "AxisymmetricFDOp",
    "create_gradient_op",
]
