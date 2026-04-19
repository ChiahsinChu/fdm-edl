# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax.numpy as jnp
import unxt

if TYPE_CHECKING:
    from typing import Sequence

    import jax

from .unit_conversion import UNIT_SYSTEMS


# todo: unit conversion
@dataclass(frozen=True)
class BoundaryCondition:
    """Boundary condition definition.

    Represents a general boundary condition of the form:

    alpha * u + beta * du/dn + gamma = 0

    Three cases:

    - Dirichlet: alpha != 0, beta = 0
    - Neumann:   alpha = 0, beta != 0
    - Robin:     alpha != 0, beta != 0

    Parameters
    ----------
    alpha : unxt.Quantity
        Coefficient multiplying the solution value u.
    beta : unxt.Quantity
        Coefficient multiplying the outward normal derivative du/dn.
    gamma : unxt.Quantity
        Constant offset term.
    node_indices : array-like of int
        Indices into the flat grid where this BC is applied.
    """

    alpha: unxt.Quantity | float
    beta: unxt.Quantity | float
    gamma: unxt.Quantity | float
    node_indices: jax.Array | Sequence[int]
    _is_dirichlet: bool = field(init=False, repr=False)

    def __post_init__(self):
        indices = jnp.asarray(self.node_indices)
        if not jnp.issubdtype(indices.dtype, jnp.integer):
            raise TypeError(f"node_indices must be integer, got {indices.dtype}")
        object.__setattr__(self, "node_indices", indices)

        #  is a unit of potential (e.g. "V" or "mV")
        if isinstance(self.alpha, unxt.Quantity):
            object.__setattr__(self, "alpha", self.alpha.to("").value)
        if isinstance(self.beta, unxt.Quantity):
            object.__setattr__(
                self, "beta", self.beta.to(UNIT_SYSTEMS["metal"]["length"]).value
            )
        if isinstance(self.gamma, unxt.Quantity):
            object.__setattr__(
                self,
                "gamma",
                self.gamma.to(UNIT_SYSTEMS["metal"]["electrical potential"]).value,
            )
        # check coefficients
        if self.alpha == 0 and self.beta == 0:
            raise ValueError("At least one of alpha or beta must be non-zero.")

        # set flag for Dirichlet BC (clamping DOFs in solver)
        if self.alpha != 0 and self.beta == 0:
            object.__setattr__(self, "_is_dirichlet", True)
        else:
            object.__setattr__(self, "_is_dirichlet", False)

    def update_residual(
        self,
        residual: jax.Array,
        phi: jax.Array,
        grad: jax.Array | None = None,
    ) -> jax.Array:
        """Compute the residual for this BC at the specified nodes.

        Parameters
        ----------
        residual : jax.Array, shape (n_grid,)
            Residual vector to be modified in-place.
        phi : jax.Array, shape (n_grid,)
            Current solution vector (potential at each node).
        grad : jax.Array, shape (n_grid,)
            Gradient of phi at each node (for Neumann/Robin BCs).

        Returns
        -------
        jax.Array, shape (n_grid,)
            Modified residual with BC constraints applied at
            ``self.node_indices``.
        """
        idx = self.node_indices
        if grad is None:
            if self.beta != 0:
                raise ValueError("grad must be provided for Neumann/Robin BCs")
            grad_term = 0.0
        else:
            grad_term = grad[idx]
        residual = residual.at[idx].set(
            self.alpha * phi[idx] + self.beta * grad_term + self.gamma
        )
        return residual

    def clamp_dirichlet(self, phi: jax.Array) -> jax.Array:
        """Clamp the solution values at Dirichlet nodes to the specified value.

        This is used in the solver to enforce Dirichlet BCs by clamping
        the DOFs at each iteration.

        Parameters
        ----------
        phi : jax.Array, shape (n_grid,)
            Current solution vector (potential at each node).

        Returns
        -------
        jax.Array, shape (n_grid,)
            Modified solution vector with Dirichlet nodes clamped to the
            specified value.
        """
        if self._is_dirichlet:
            phi = phi.at[self.node_indices].set(-self.gamma / self.alpha)
        return phi

    @property
    def is_dirichlet(self) -> bool:
        """Whether this boundary condition is Dirichlet."""
        return self._is_dirichlet
