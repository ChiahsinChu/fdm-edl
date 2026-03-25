# SPDX-License-Identifier: GPL-3.0-or-later
"""Robin (mixed) boundary condition."""

from __future__ import annotations

import jax.numpy as raw_jnp
import unxt

from .base import BoundaryCondition


class RobinBC(BoundaryCondition):
    """Robin (mixed) boundary condition: α·φ + β·∂φ/∂n = g.

    Combines Dirichlet and Neumann contributions.  Setting β = 0
    recovers a Dirichlet BC; setting α = 0 recovers a Neumann BC.

    Parameters
    ----------
    node_indices : array-like of int
        Indices of boundary nodes.
    alpha : float or array-like
        Coefficient of the Dirichlet term.
    beta : float or array-like
        Coefficient of the Neumann (flux) term.
    g_values : unxt.Quantity
        Right-hand-side values of the Robin condition.
    neighbor_indices : array-like of int
        Interior neighbor indices for the one-sided FD flux estimate.
    """

    def __init__(
        self,
        node_indices,
        alpha,
        beta,
        g_values: unxt.Quantity,
        neighbor_indices,
    ):
        super().__init__(node_indices)
        self.alpha = alpha
        self.beta = beta
        self.g_values = g_values
        self.neighbor_indices = raw_jnp.asarray(neighbor_indices)

    def apply_residual(self, residual, phi, coordinates):
        """Enforce α·φ + β·∂φ/∂n = g.

        Parameters
        ----------
        residual : unxt.Quantity, shape (n_grid,)
            Physics residual at every grid node.
        phi : unxt.Quantity, shape (n_grid,)
            Current potential at every grid node.
        coordinates : unxt.Quantity
            Grid coordinates (1-D).

        Returns
        -------
        unxt.Quantity, shape (n_grid,)
            Residual with Robin constraints applied at
            ``self.node_indices``.
        """
        x_bc = coordinates[self.node_indices]
        x_nb = coordinates[self.neighbor_indices]
        dx = x_bc - x_nb

        phi_bc = phi[self.node_indices]
        phi_nb = phi[self.neighbor_indices]
        fd_deriv = (phi_bc - phi_nb) / dx

        bc_value = (self.alpha * phi_bc + self.beta * fd_deriv - self.g_values).value
        bc_residual = unxt.Quantity(bc_value, unit=residual.unit)
        return residual.at[self.node_indices].set(bc_residual)
