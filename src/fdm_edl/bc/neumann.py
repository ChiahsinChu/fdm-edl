# SPDX-License-Identifier: GPL-3.0-or-later
"""Neumann (fixed-flux) boundary condition."""

from __future__ import annotations

import jax.numpy as raw_jnp
import unxt

from .base import BoundaryCondition


class NeumannBC(BoundaryCondition):
    """Fixed-flux (Neumann) boundary condition: ∂φ/∂n = g at given nodes.

    Uses a one-sided finite-difference approximation to enforce the
    prescribed normal derivative.  For 1-D problems the normal direction
    is inferred automatically (leftmost node → -x, rightmost → +x).

    Parameters
    ----------
    node_indices : array-like of int
        Indices into the flat grid for this BC.
    flux_values : unxt.Quantity
        Prescribed outward normal flux ∂φ/∂n at each boundary node.
        Must be broadcastable to ``len(node_indices)``.
    neighbor_indices : array-like of int
        For each boundary node, the index of the nearest interior neighbor
        used in the one-sided FD stencil.
    """

    def __init__(
        self,
        node_indices,
        flux_values: unxt.Quantity,
        neighbor_indices,
    ):
        super().__init__(node_indices)
        self.flux_values = flux_values
        self.neighbor_indices = raw_jnp.asarray(neighbor_indices)

    def apply_residual(self, residual, phi, coordinates):
        """Enforce ∂φ/∂n = g via one-sided finite difference.

        For each boundary node *i* with interior neighbor *j*, the
        residual is set to::

            (φ[i] - φ[j]) / |x[i] - x[j]| · sign - flux_value

        where ``sign`` encodes the outward normal direction.
        """
        x_bc = coordinates[self.node_indices]
        x_nb = coordinates[self.neighbor_indices]
        dx = x_bc - x_nb  # signed distance (carries outward-normal sign)

        # one-sided derivative: (phi_bc - phi_nb) / dx  = g
        phi_bc = phi[self.node_indices]
        phi_nb = phi[self.neighbor_indices]
        fd_deriv = (phi_bc - phi_nb) / dx

        bc_value = (fd_deriv - self.flux_values).to("V/m").value
        bc_residual = unxt.Quantity(bc_value, unit=residual.unit)
        return residual.at[self.node_indices].set(bc_residual)
