# SPDX-License-Identifier: GPL-3.0-or-later
"""Boundary condition base class and Dirichlet BC implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod

import jax.numpy as raw_jnp
import unxt


class BoundaryCondition(ABC):
    """Abstract base class for boundary conditions.

    A boundary condition acts on a subset of grid nodes identified by
    ``node_indices``.  Subclasses must implement :meth:`apply_residual`
    to modify the global residual vector at those nodes, enforcing the
    desired constraint.

    Parameters
    ----------
    node_indices : array-like of int
        Indices into the flat grid identifying this BC's nodes.
    """

    def __init__(self, node_indices):
        self.node_indices = raw_jnp.asarray(node_indices)

    @abstractmethod
    def apply_residual(
        self,
        residual: unxt.Quantity,
        phi: unxt.Quantity,
        coordinates: unxt.Quantity,
    ) -> unxt.Quantity:
        """Modify *residual* in-place at this BC's nodes.

        Parameters
        ----------
        residual : unxt.Quantity, shape (n_grid,)
            Physics residual at every grid node.
        phi : unxt.Quantity, shape (n_grid,)
            Current potential at every grid node.
        coordinates : unxt.Quantity, shape (n_grid,) or (n_grid, n_dim)
            Grid coordinates.

        Returns
        -------
        unxt.Quantity, shape (n_grid,)
            Modified residual with BC constraints applied at
            ``self.node_indices``.
        """
        ...

    @abstractmethod
    def compute_violation(
        self,
        phi: unxt.Quantity,
        coordinates: unxt.Quantity,
    ) -> unxt.Quantity:
        """Compute the BC violation at this BC's nodes.

        Returns the constraint error (in the BC's natural units) at
        the boundary nodes only, without modifying the global residual.
        Used by the PINN-style penalty loss in
        :class:`~fdm_edl.solver.OptaxSolver`.

        Parameters
        ----------
        phi : unxt.Quantity, shape (n_grid,)
            Current potential at every grid node.
        coordinates : unxt.Quantity, shape (n_grid,) or (n_grid, n_dim)
            Grid coordinates.

        Returns
        -------
        unxt.Quantity, shape (n_bc_nodes,)
            Constraint violation at each boundary node.
        """
        ...

    def apply_initial_guess(self, phi0: unxt.Quantity) -> unxt.Quantity:
        """Optionally set initial values at BC nodes.

        The default implementation returns *phi0* unchanged.  Subclasses
        (e.g. Dirichlet) may override this to seed the initial guess.

        Parameters
        ----------
        phi0 : unxt.Quantity, shape (n_grid,)
            Initial guess for the potential at all grid nodes.

        Returns
        -------
        unxt.Quantity, shape (n_grid,)
            Potentially modified initial guess.
        """
        return phi0


class DirichletBC(BoundaryCondition):
    """Fixed-value (Dirichlet) boundary condition: φ = φ₀ at given nodes.

    Parameters
    ----------
    node_indices : array-like of int
        Indices into the flat grid for this BC.
    values : unxt.Quantity
        Prescribed potential at each boundary node.  Must be
        broadcastable to the number of nodes in *node_indices*.
    """

    def __init__(self, node_indices, values: unxt.Quantity):
        super().__init__(node_indices)
        self.values = values

    def apply_residual(self, residual, phi, coordinates):
        """Enforce φ[i] = value by setting residual[i] = φ[i] - value.

        The numerical difference ``(φ - φ₀)`` (in Volts) is placed into
        the residual with the same unit tag, so the Jacobian stays
        dimensionally consistent.

        Parameters
        ----------
        residual : unxt.Quantity, shape (n_grid,)
            Physics residual at every grid node.
        phi : unxt.Quantity, shape (n_grid,)
            Current potential at every grid node.
        coordinates : unxt.Quantity
            Grid coordinates (unused by Dirichlet, accepted for API
            compatibility).

        Returns
        -------
        unxt.Quantity, shape (n_grid,)
            Residual with Dirichlet constraints applied at
            ``self.node_indices``.
        """
        bc_value = (phi[self.node_indices] - self.values).to("V").value
        bc_residual = unxt.Quantity(bc_value, unit=residual.unit)
        return residual.at[self.node_indices].set(bc_residual)

    def compute_violation(self, phi, coordinates):
        """Return φ[i] - φ₀ at the Dirichlet nodes."""
        return (phi[self.node_indices] - self.values).to("V")

    def apply_initial_guess(self, phi0):
        """Seed the initial guess with the prescribed Dirichlet values.

        Parameters
        ----------
        phi0 : unxt.Quantity, shape (n_grid,)
            Initial guess for the potential at all grid nodes.

        Returns
        -------
        unxt.Quantity, shape (n_grid,)
            Modified initial guess with Dirichlet values inserted at
            ``self.node_indices``.
        """
        values_in_unit = unxt.Quantity(self.values.to("V").value, unit=phi0.unit)
        return phi0.at[self.node_indices].set(values_in_unit)
