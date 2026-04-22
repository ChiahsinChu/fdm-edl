# SPDX-License-Identifier: GPL-3.0-or-later
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ...utils import constants
from ...utils import unit_conversion as uc
from .base import BaseGradientOP, _nonuniform_derivative

Array = jax.Array


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LaplacianOP(BaseGradientOP):
    """
    Numerical 1D Laplacian operator for sampled data ``y ≈ f(x)``.

    The operator computes the second derivative ``d²y/dx²`` (the 1D Laplacian) from samples on a grid ``x``.

    This class is designed to be JIT-friendly in JAX:
    - The instance contains only small Python scalars (no arrays).
    - The instance is registered as a PyTree with *static* auxiliary data, so you can do
      ``op_jit = jax.jit(op)`` and call it as ``op_jit(x, y, h=...)``.

    Parameters
    ----------
    uniform : bool, default=True
        If True, assumes (approximately) uniform spacing and uses fixed-coefficient stencils.
        If False, uses local finite-difference weights (Fornberg) that handle nonuniform grids.
    boundary_points : {3, 4, 5}, default=4
        Number of points used in the one-sided stencils at the boundaries.

        For ``uniform=True``, larger values improve boundary accuracy:
        - 3-point: gradient is 2nd-order at the boundary; Laplacian is 1st-order at the boundary
        - 4-point: gradient is 3rd-order at the boundary; Laplacian is 2nd-order at the boundary
        - 5-point: gradient is 4th-order at the boundary; Laplacian is 3rd-order at the boundary

        For ``uniform=False``, this controls the number of nodes used to compute nonuniform
        one-sided weights at the boundaries.
    interior_points : {3, 5}, default=3
        Number of points used in the centered interior stencil.

        For ``uniform=True``:
        - 3-point centered stencil is 2nd-order accurate in the interior.
        - 5-point centered stencil is 4th-order accurate in the interior (with a small fallback
          used for the two points adjacent to the boundaries).

        For ``uniform=False``:
        - Uses a centered window of size 3 or 5 to compute nonuniform weights per interior point.

    Notes
    -----
    - ``x`` must be 1D and sorted ascending.
    - In JAX, changing array shapes (length of ``x``/``y``) triggers recompilation; keep shapes fixed
      for best performance.
    - In uniform mode, for maximum stability under `jit`, consider passing ``h`` explicitly
      (e.g., ``h = x[1] - x[0]``) rather than relying on computing ``mean(diff(x))`` inside the jitted function.

    Examples
    --------
    Uniform grid, 5-point interior and boundary stencils::

        import jax
        import jax.numpy as jnp
        from fdm_edl.utils.grad import GradLaplacian1D

        x = jnp.linspace(0.0, 1.0, 1024)
        y = jnp.sin(2 * jnp.pi * x)

        op = GradLaplacian1D(uniform=True, boundary_points=5, interior_points=5)
        op_jit = jax.jit(op)

        h = x[1] - x[0]
        grad, lap = op_jit(x, y, h=h)

    Nonuniform grid::

        x = jnp.sort(jax.random.uniform(jax.random.key(0), (1024,)))
        y = jnp.sin(2 * jnp.pi * x)

        op = GradLaplacian1D(uniform=False, boundary_points=5, interior_points=5)
        grad, lap = jax.jit(op)(x, y)
    """

    def _div_D(self, x: Array, y: Array, grad: Array) -> Array:
        """
        Compute Laplacian (2nd derivative) only.

        Parameters
        ----------
        x : jax.numpy.ndarray
            Grid locations with shape ``(n,)``.
        y : jax.numpy.ndarray
            Function values with shape ``(n,)``.

        Returns
        -------
        lap : jax.numpy.ndarray
            Second derivative approximation with shape ``(n,)``.
        """

        if self.uniform:
            lap = _uniform_lap(x, y, self.boundary_points, self.interior_points)
        else:
            lap = _nonuniform_lap(x, y, self.boundary_points, self.interior_points)

        eps_0 = constants.VACUUM_PERMITTIVITY.to(
            uc.UNIT_SYSTEMS["metal"]["permittivity"]
        ).value
        if self.eps_func is None:
            eps = eps_0
        else:
            eps = self.eps_func(jnp.abs(grad)) * eps_0
        return -lap * eps


def _uniform_lap(
    x: Array, y: Array, boundary_points: int, interior_points: int
) -> Array:
    """
    Uniform-grid backend for Laplacian (2nd derivative).
    """
    hh = jnp.mean(x[1:] - x[:-1])
    hh2 = hh * hh

    lap = jnp.empty_like(y)

    # --- interior ---
    if interior_points == 3:
        lap = lap.at[1:-1].set((y[:-2] - 2 * y[1:-1] + y[2:]) / hh2)
    else:
        lap = lap.at[2:-2].set(
            (-y[4:] + 16 * y[3:-1] - 30 * y[2:-2] + 16 * y[1:-3] - y[:-4]) / (12 * hh2)
        )

        # fallback for i=1 and i=n-2
        lap = lap.at[1].set((y[0] - 2 * y[1] + y[2]) / hh2)
        lap = lap.at[-2].set((y[-3] - 2 * y[-2] + y[-1]) / hh2)

    # --- boundaries ---
    k = boundary_points
    if k == 3:
        lap = lap.at[0].set((y[0] - 2 * y[1] + y[2]) / hh2)
        lap = lap.at[-1].set((y[-3] - 2 * y[-2] + y[-1]) / hh2)
    elif k == 4:
        lap = lap.at[0].set((2 * y[0] - 5 * y[1] + 4 * y[2] - y[3]) / hh2)
        lap = lap.at[-1].set((2 * y[-1] - 5 * y[-2] + 4 * y[-3] - y[-4]) / hh2)
    else:  # k == 5
        lap = lap.at[0].set(
            (35 * y[0] - 104 * y[1] + 114 * y[2] - 56 * y[3] + 11 * y[4]) / (12 * hh2)
        )
        lap = lap.at[-1].set(
            (35 * y[-1] - 104 * y[-2] + 114 * y[-3] - 56 * y[-4] + 11 * y[-5])
            / (12 * hh2)
        )

    return lap


def _nonuniform_lap(
    x: Array, y: Array, boundary_points: int, interior_points: int
) -> Array:
    """
    Nonuniform-grid backend for Laplacian (2nd derivative).
    """
    return _nonuniform_derivative(x, y, boundary_points, interior_points, order=2)
