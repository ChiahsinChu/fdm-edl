# SPDX-License-Identifier: GPL-3.0-or-later
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..base import BaseGradientOP

Array = jax.Array


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class EuclideanFDOp(BaseGradientOP):
    """
        Finite-difference 1-D gradient and displacement-divergence operator.

        The operator returns the first derivative ``dy/dx`` together with
        ``-ε d²y/dx²`` evaluated from samples on a grid ``x``. When ``eps_func`` is
        provided, the permittivity is evaluated from ``|dy/dx|`` before forming the
        second output.

    This class is designed to be JIT-friendly in JAX:
    - The instance contains only small Python scalars (no arrays).
    - The instance is registered as a PyTree with *static* auxiliary data, so you can do
            ``op_jit = jax.jit(op)`` and call it as ``op_jit(x, y)``.

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
        from fdm_edl.solver.grad import EuclideanFDOp

        x = jnp.linspace(0.0, 1.0, 1024)
        y = jnp.sin(2 * jnp.pi * x)

        op = EuclideanFDOp(uniform=True, boundary_points=5, interior_points=5)
        op_jit = jax.jit(op)
        grad, div_D = op_jit(x, y)

    Nonuniform grid::

        x = jnp.sort(jax.random.uniform(jax.random.key(0), (1024,)))
        y = jnp.sin(2 * jnp.pi * x)

        op = EuclideanFDOp(uniform=False, boundary_points=5, interior_points=5)
        grad, div_D = jax.jit(op)(x, y)
    """

    def _grad(self, x: Array, y: Array) -> Array:
        """
        Compute gradient (1st derivative).

        Parameters
        ----------
        x : jax.numpy.ndarray
            Grid locations with shape ``(n,)``.
        y : jax.numpy.ndarray
            Function values with shape ``(n,)``.

        Returns
        -------
        grad : jax.numpy.ndarray
            First derivative approximation with shape ``(n,)``.
        """
        if self.uniform:
            return _uniform_grad(x, y, self.boundary_points, self.interior_points)
        return _nonuniform_grad(x, y, self.boundary_points, self.interior_points)

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
        lap = self._lap(x, y)
        # relative permittivity as a function of |E|=|grad|
        eps = self.eps_func(jnp.abs(grad))
        return -lap * eps

    def _lap(self, x: Array, y: Array) -> Array:
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
        return lap


def _fd_coeffs_1d(x_nodes: Array, x0: Array | float, deriv_order: int) -> Array:
    """
    Compute finite-difference weights on (possibly) nonuniform nodes.

    This uses the Fornberg algorithm to compute weights ``w`` such that::

        f^(m)(x0) ≈ sum_j w[j] * f(x_nodes[j])

    Parameters
    ----------
    x_nodes : jax.numpy.ndarray
        Array of node locations with shape ``(k,)``.
    x0 : float
        Evaluation point.
    deriv_order : int
        Derivative order ``m`` (e.g., 1 for first derivative, 2 for second derivative).

    Returns
    -------
    w : jax.numpy.ndarray
        Weights with shape ``(k,)``.

    Notes
    -----
    - Requires ``k >= deriv_order + 1``.
    - Works for nonuniform spacing.
    """
    n = x_nodes.shape[0]
    m = int(deriv_order)
    c = jnp.zeros((n, m + 1), dtype=x_nodes.dtype).at[0, 0].set(1.0)
    c1 = 1.0
    c4 = x_nodes[0] - x0

    def outer_body(i, state):
        c, c1, c4 = state
        c2 = 1.0
        c5 = c4
        c4 = x_nodes[i] - x0

        def inner_body(j, inner_state):
            c, c2 = inner_state
            c3 = x_nodes[i] - x_nodes[j]
            c2 = c2 * c3

            def k_body(k, row):
                val = jnp.where(
                    k == 0,
                    (c4 * row[0]) / c3,
                    (c4 * row[k] - k * row[k - 1]) / c3,
                )
                return row.at[k].set(val)

            rowj = jax.lax.fori_loop(0, m + 1, k_body, c[j, :])
            c = c.at[j, :].set(rowj)
            return (c, c2)

        (c, c2) = jax.lax.fori_loop(0, i, inner_body, (c, c2))

        def k_body_i(k, row):
            val = jnp.where(
                k == 0,
                (-c1 * c5 * c[i - 1, 0]) / c2,
                (c1 * (k * c[i - 1, k - 1] - c5 * c[i - 1, k])) / c2,
            )
            return row.at[k].set(val)

        rowi = jax.lax.fori_loop(0, m + 1, k_body_i, c[i, :])
        c = c.at[i, :].set(rowi)

        c1 = c2
        return (c, c1, c4)

    (c, _, _) = jax.lax.fori_loop(1, n, outer_body, (c, c1, c4))
    return c[:, m]


def _uniform_grad(
    x: Array, y: Array, boundary_points: int, interior_points: int
) -> Array:
    """
    Uniform-grid backend for gradient (1st derivative).
    """
    hh = jnp.mean(x[1:] - x[:-1])

    grad = jnp.empty_like(y)

    # --- interior ---
    if interior_points == 3:
        grad = grad.at[1:-1].set((y[2:] - y[:-2]) / (2 * hh))
    else:
        grad = grad.at[2:-2].set(
            (-y[4:] + 8 * y[3:-1] - 8 * y[1:-3] + y[:-4]) / (12 * hh)
        )

        # fallback for i=1 and i=n-2
        grad = grad.at[1].set((y[2] - y[0]) / (2 * hh))
        grad = grad.at[-2].set((y[-1] - y[-3]) / (2 * hh))

    # --- boundaries ---
    k = boundary_points
    if k == 3:
        grad = grad.at[0].set((-3 * y[0] + 4 * y[1] - y[2]) / (2 * hh))
        grad = grad.at[-1].set((3 * y[-1] - 4 * y[-2] + y[-3]) / (2 * hh))
    elif k == 4:
        grad = grad.at[0].set((-11 * y[0] + 18 * y[1] - 9 * y[2] + 2 * y[3]) / (6 * hh))
        grad = grad.at[-1].set(
            (11 * y[-1] - 18 * y[-2] + 9 * y[-3] - 2 * y[-4]) / (6 * hh)
        )
    else:  # k == 5
        grad = grad.at[0].set(
            (-25 * y[0] + 48 * y[1] - 36 * y[2] + 16 * y[3] - 3 * y[4]) / (12 * hh)
        )
        grad = grad.at[-1].set(
            (25 * y[-1] - 48 * y[-2] + 36 * y[-3] - 16 * y[-4] + 3 * y[-5]) / (12 * hh)
        )
    return grad


def _nonuniform_derivative(
    x: Array, y: Array, boundary_points: int, interior_points: int, order: int
) -> Array:
    """
    Nonuniform-grid backend for a single derivative order via local Fornberg weights.
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    n = x.shape[0]
    kB = boundary_points
    kI = interior_points
    r = (kI - 1) // 2  # 1 (3-pt) or 2 (5-pt)

    out = jnp.empty_like(y)

    def onesided_eval(i: int, order: int, left_bias: bool) -> Array:
        idx = jnp.arange(0, kB) if left_bias else jnp.arange(n - kB, n)
        x_nodes = x[idx]
        y_nodes = y[idx]
        w = _fd_coeffs_1d(x_nodes, x[i], order)
        return jnp.sum(w * y_nodes)

    # endpoints
    out = out.at[0].set(onesided_eval(0, order, True))
    out = out.at[-1].set(onesided_eval(n - 1, order, False))

    def centered_eval(i: int, order: int) -> Array:
        idx = jnp.arange(i - r, i + r + 1)
        x_nodes = x[idx]
        y_nodes = y[idx]
        w = _fd_coeffs_1d(x_nodes, x[i], order)
        return jnp.sum(w * y_nodes)

    # indices where centered window fits
    ii = jnp.arange(r, n - r)
    out = out.at[ii].set(jax.vmap(lambda i: centered_eval(i, order))(ii))

    # near-boundary points not covered by centered stencil (only when r=2, i.e., 5-pt interior)
    if kI == 5:
        out = out.at[1].set(onesided_eval(1, order, True))
        out = out.at[-2].set(onesided_eval(n - 2, order, False))

    return out


def _nonuniform_grad(
    x: Array, y: Array, boundary_points: int, interior_points: int
) -> Array:
    """
    Nonuniform-grid backend for gradient (1st derivative).
    """
    return _nonuniform_derivative(x, y, boundary_points, interior_points, order=1)


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
