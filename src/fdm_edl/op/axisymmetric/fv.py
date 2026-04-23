# SPDX-License-Identifier: GPL-3.0-or-later
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..cartesian.fv import EuclideanFVOp, _faces_from_centers

Array = jax.Array


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AxisymmetricFVOp(EuclideanFVOp):
    def _div_D(self, x: Array, y: Array, grad: Array) -> Array:
        xf = _faces_from_centers(x)  # (N+1,)
        dx_face = x[1:] - x[:-1]  # (N-1,) δx_{i+1/2}

        # Face-centred positive gradient and electric field
        grad_face = (y[1:] - y[:-1]) / dx_face  # (N-1,)  dy/dx at faces
        # grad_face = grad

        # Displacement flux D = ε(|E|)·E at faces
        dfield_face = -self.eps_func(jnp.abs(grad_face)) * grad_face  # (N-1,)

        # Axisymmetric cylindrical radial divergence:
        # div(D) = (1/r) d(r D_r)/dr, written as flux balance in annular control volumes.
        r_face = xf[1:-1]  # (N-1,) interior face radii r_{i+1/2}
        area_flux_right = r_face[1:] * dfield_face[1:]
        area_flux_left = r_face[:-1] * dfield_face[:-1]
        vol_interior = 0.5 * (xf[2:-1] ** 2 - xf[1:-2] ** 2)
        div_interior = (area_flux_right - area_flux_left) / vol_interior  # (N-2,)

        # Pad to (N,): boundary rows replicate nearest interior value.
        # These are overwritten by the caller when applying BCs.
        div_dfield = jnp.concatenate(
            [div_interior[:1], div_interior, div_interior[-1:]]
        )  # (N,)
        return div_dfield
