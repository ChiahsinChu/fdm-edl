# SPDX-License-Identifier: GPL-3.0-or-later
"""Periodic boundary condition."""

from __future__ import annotations

import jax.numpy as raw_jnp
import unxt

from .base import BoundaryCondition


class PeriodicBC(BoundaryCondition):
    """Periodic boundary condition: φ(x_left) = φ(x_right).

    Enforces that the potential at paired left and right boundary nodes
    is equal.  The ``node_indices`` are taken as the *left* side; the
    *right* side is given by ``partner_indices``.

    Parameters
    ----------
    node_indices : array-like of int
        Indices of the "left" periodic boundary nodes.
    partner_indices : array-like of int
        Indices of the corresponding "right" periodic boundary nodes.
        Must have the same length as *node_indices*.
    """

    def __init__(self, node_indices, partner_indices):
        super().__init__(node_indices)
        self.partner_indices = raw_jnp.asarray(partner_indices)

    def apply_residual(self, residual, phi, coordinates):
        """Enforce φ[left] - φ[right] = 0 at the left nodes.

        The residual at the left boundary nodes is set to
        ``φ[left] - φ[right]``.  The right partner nodes keep their
        physics residual (they are interior-like after the periodic
        coupling).
        """
        bc_value = (phi[self.node_indices] - phi[self.partner_indices]).to("V").value
        bc_residual = unxt.Quantity(bc_value, unit=residual.unit)
        return residual.at[self.node_indices].set(bc_residual)

    def compute_violation(self, phi, coordinates):
        """Return φ[left] - φ[right] at the periodic nodes."""
        return (phi[self.node_indices] - phi[self.partner_indices]).to("V")
