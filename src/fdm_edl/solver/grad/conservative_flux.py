# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .base import BaseGradientOP

Array = jax.Array


def _unit_eps(efield: Array) -> Array:
    """Return unit relative permittivity for all field values."""
    return jnp.ones_like(efield)


# ---------------------------------------------------------------------------
# Private pure-JAX helpers (no Python loops; fully JIT-compatible)
# ---------------------------------------------------------------------------


def _faces_from_centers(x: Array) -> Array:
    """
    Compute face locations ``x_{i+1/2}`` from cell-centre positions ``x_i``.

    Returns an array of shape ``(N+1,)`` for input of shape ``(N,)``.
    Interior faces are midpoints of adjacent centres; boundary faces are
    linearly extrapolated.
    """
    xf_int = 0.5 * (x[:-1] + x[1:])  # (N-1,)
    xf_left = x[0] - 0.5 * (x[1] - x[0])
    xf_right = x[-1] + 0.5 * (x[-1] - x[-2])
    return jnp.concatenate([xf_left[None], xf_int, xf_right[None]])  # (N+1,)


def _interpolate_faces_to_cells(G_face: Array) -> Array:
    """
    Average face-centred values ``G_{i+1/2}`` (shape ``(N-1,)``) to cell centres.

    Returns shape ``(N,)``:
    - Interior cells (1 .. N-2): arithmetic mean of the two neighbouring faces.
    - Boundary cells (0, N-1): one-sided (nearest face value).
    """
    G_int = 0.5 * (G_face[:-1] + G_face[1:])  # (N-2,)
    return jnp.concatenate([G_face[:1], G_int, G_face[-1:]])  # (N,)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ConservativeFluxGradOP(BaseGradientOP):
    """
    Conservative finite-volume gradient operator for sampled data ``y ≈ f(x)``.

    Implements a cell-centred finite-volume scheme that computes both the
    gradient ``dy/dx`` and the divergence of the displacement field
    ``∇·(ε E) = ∇·(ε (−∇y))`` in a flux-conservative manner on a
    (possibly nonuniform) 1-D grid.

    This class is designed to be JIT-friendly in JAX:
    - The instance contains only small Python scalars and a callable (no arrays).
    - The instance is registered as a PyTree with *static* auxiliary data, so you can do
      ``op_jit = jax.jit(op)`` and call it as ``op_jit(x, y)``.

    Parameters
    ----------
    uniform : bool, default=True
        Inherited from :class:`BaseGradientOP`. **Not used** by this operator
        — the FV scheme always derives distances from the actual grid.
    boundary_points : {3, 4, 5}, default=4
        Inherited from :class:`BaseGradientOP`. **Not used** by this operator.
    interior_points : {3, 5}, default=3
        Inherited from :class:`BaseGradientOP`. **Not used** by this operator.
    eps_func : callable, default=`_unit_eps`
        Permittivity as a function of the electric field ``E``:
        ``ε = eps_func(E)`` where ``E`` has shape ``(N-1,)``.
        Defaults to ``ε = 1`` (constant, linear dielectric).
        Stored as static aux data — changing ``eps_func`` triggers JAX recompilation.

    Notes
    -----
    - ``x`` must be 1-D and sorted ascending.
    - Interior divergence rows are exact FV divergences; boundary rows
      replicate the nearest interior value and should be overwritten by the
      caller when applying Dirichlet boundary conditions.

    Examples
    --------
    Default (constant permittivity)::

        import jax.numpy as jnp
        from fdm_edl.solver.grad import ConservativeFluxGradOP

        x = jnp.linspace(0.0, 1.0, 256)
        phi = jnp.sin(2 * jnp.pi * x)

        op = ConservativeFluxGradOP()
        grad, div_D = op(x, phi)

    Nonlinear permittivity::

        def my_eps(E):
            return 1.0 + 0.5 * (E / 1.0) ** 2

        op = ConservativeFluxGradOP(eps_func=my_eps)
        grad, div_D = jax.jit(op)(x, phi)
    """

    eps_func: Callable[[Array], Array] = _unit_eps

    # --- public API ---
    def __call__(self, x: Array, y: Array) -> tuple[Array, Array]:
        """
        Compute cell-centred gradient and divergence of the displacement field.

        Parameters
        ----------
        x : Array, shape (N,)
            Grid locations (cell centres). Need not be sorted — they are
            sorted internally before computation.
        y : Array, shape (N,)
            Field values at cell centres, ``y ≈ f(x)``.

        Returns
        -------
        G_cell : Array, shape (N,)
            First derivative ``dy/dx`` averaged to cell centres.
        div_dfield : Array, shape (N,)
            Divergence of displacement field ``∇·(ε E) = ∇·(ε (−∇y))``.
            Boundary rows replicate the nearest interior value.
        """
        grad, _ = super().__call__(x, y)

        sorted_idx = jnp.argsort(x)
        _x = x[sorted_idx]
        _y = y[sorted_idx]
        div_dfield = self._div_dfield(_x, _y)
        return (grad, div_dfield)

    def _div_dfield(self, x: Array, y: Array) -> Array:
        """
        Compute divergence of the displacement field ``∇·(ε E)`` only.

        Parameters
        ----------
        x : Array, shape (N,)
            Sorted grid locations.
        y : Array, shape (N,)
            Field values at cell centres.

        Returns
        -------
        div_dfield : Array, shape (N,)
            ``∇·(ε (−∇y))`` at cell centres.
            Boundary rows replicate the nearest interior value.
        """
        # _, div = _conservative_flux(x, y, self.eps_func)
        xf = _faces_from_centers(x)  # (N+1,)
        dx_cell = xf[1:] - xf[:-1]  # (N,)   Δx_i
        dx_face = x[1:] - x[:-1]  # (N-1,) δx_{i+1/2}

        # Face-centred positive gradient and electric field
        grad_face = (y[1:] - y[:-1]) / dx_face  # (N-1,)  dy/dx at faces

        # Displacement flux D = ε(|E|)·E at faces
        dfield_face = -self.eps_func(jnp.abs(grad_face)) * grad_face  # (N-1,)

        # # Cell-centred gradient (conservative average of adjacent face values)
        # grad_cell = _interpolate_faces_to_cells(grad_face)  # (N,)

        # Interior divergence: ∇·D at nodes 1 .. N-2
        div_interior = (dfield_face[1:] - dfield_face[:-1]) / dx_cell[1:-1]  # (N-2,)

        # Pad to (N,): boundary rows replicate nearest interior value.
        # These are overwritten by the caller when applying BCs.
        div_dfield = jnp.concatenate(
            [div_interior[:1], div_interior, div_interior[-1:]]
        )  # (N,)
        return div_dfield
