# SPDX-License-Identifier: GPL-3.0-or-later
"""Discrete differential operators for FDM-EDL."""

from __future__ import annotations

from abc import ABC, abstractmethod

import quaxed.numpy as jnp
import unxt


class LaplacianOperator(ABC):
    """Abstract Laplacian operator: maps φ → ∇²φ on a discrete grid.

    Subclasses implement the stencil for a specific dimensionality and
    mesh topology.
    """

    @abstractmethod
    def __call__(self, phi: unxt.Quantity) -> unxt.Quantity:
        """Evaluate ∇²φ at every grid node.

        Parameters
        ----------
        phi : unxt.Quantity, shape (n_grid,)
            Potential values at all grid nodes.

        Returns
        -------
        unxt.Quantity, shape (n_grid,)
            Laplacian ∇²φ at each node.  Values at boundary nodes are
            typically meaningless (overwritten by BCs).
        """
        ...


class LaplacianOperator1D(LaplacianOperator):
    """1-D Laplacian using a 3-point central finite-difference stencil
    on a nonuniform grid.

    The stencil is precomputed from the grid coordinates so that each
    call only performs array arithmetic (no recomputation of spacings).

    Parameters
    ----------
    coordinates : unxt.Quantity, shape (n_grid,)
        1-D grid coordinates in physical units.  Must be monotonically
        increasing and have at least 3 nodes.
    """

    def __init__(self, coordinates: unxt.Quantity):
        x = coordinates.to("m")
        # Precompute spacings for interior nodes (indices 1 .. N-2).
        self._dx_left = x[1:-1] - x[:-2]
        self._dx_right = x[2:] - x[1:-1]
        self._coeff = 2.0 / (self._dx_left + self._dx_right)
        self._n_grid = coordinates.shape[0]

    def __call__(self, phi: unxt.Quantity) -> unxt.Quantity:
        """Evaluate d²φ/dx² at interior nodes; zero at boundaries."""
        phi_m = phi.to("V")

        d2phi_interior = self._coeff * (
            (phi_m[2:] - phi_m[1:-1]) / self._dx_right
            - (phi_m[1:-1] - phi_m[:-2]) / self._dx_left
        )

        # Pad boundary positions with zeros (BCs will overwrite).
        zero = unxt.Quantity(jnp.zeros((1,)), unit=d2phi_interior.unit)
        return jnp.concatenate([zero, d2phi_interior, zero])
