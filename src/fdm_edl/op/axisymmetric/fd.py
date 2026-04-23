# SPDX-License-Identifier: GPL-3.0-or-later
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..cartesian.fd import EuclideanFDOp

Array = jax.Array


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AxisymmetricFDOp(EuclideanFDOp):
    def _div_D(self, x: Array, y: Array, grad: Array) -> Array:
        lap = self._lap(x, y)
        # Axisymmetric cylindrical radial operator: d2/dr2 + (1/r) d/dr.
        # At r=0, use the regularity limit (d/dr)/r -> d2/dr2.
        radial_term = jnp.where(jnp.abs(x) > jnp.finfo(x.dtype).eps, grad / x, lap)
        lap_op = lap + radial_term
        # relative permittivity as a function of |E|=|grad|
        eps = self.eps_func(jnp.abs(grad))
        return -lap_op * eps
