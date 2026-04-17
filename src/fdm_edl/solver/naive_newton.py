# SPDX-License-Identifier: GPL-3.0-or-later
"""Nonlinear root-finding solvers for the FDM-EDL system."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
from jax import numpy as jnp

if TYPE_CHECKING:
    from typing import Sequence

from ..utils.bc import BoundaryCondition
from .base import BaseSolver, ResidualFunction, RootSolveResult


class NewtonSolver(BaseSolver, methods=("newton",)):
    """
    Newton-Raphson solver using JAX automatic differentiation for the
    Jacobian.

    Parameters
    ----------
    method : str or None, optional
        Ignored; present for API compatibility with the factory.
    max_iter : int, optional
        Maximum number of Newton iterations (default: 15).
    tol : float, optional
        Convergence tolerance on the step norm (default: 1e-6).

    Attributes
    ----------
    max_iter : int
        Maximum iteration count.
    tol : float
        Step-norm convergence threshold.
    """

    def __init__(
        self,
        method: str | None = None,
        max_iter: int = 20,
        tol: float = 1e-6,
    ):
        self.max_iter = max_iter
        self.tol = tol

    def solve(
        self,
        residual_fn: ResidualFunction,
        phi: jax.Array,
        boundary_conditions: Sequence[BoundaryCondition],
        *args,
    ) -> RootSolveResult:
        """
        Solve via Newton-Raphson iterations with a full JAX Jacobian.

        The iteration loop uses ``jax.lax.while_loop`` so the entire
        solve is JIT-compilable.  ``while_loop`` is preferred over
        ``fori_loop`` because it supports early termination on
        convergence, avoiding unnecessary Jacobian evaluations.

        Parameters
        ----------
        residual_fn : callable
            Residual function with signature
            ``residual_fn(phi, *args) -> jax.Array``.
        phi : jax.Array
            Initial guess.
        boundary_conditions : Sequence[BoundaryCondition]
            List of boundary conditions.
        *args
            Extra positional arguments forwarded to *residual_fn*.

        Returns
        -------
        RootSolveResult
            Result containing the solution and solver diagnostics.
        """
        max_iter = self.max_iter
        tol = self.tol

        # clamp Dirichlet nodes in the initial guess
        for bc in boundary_conditions:
            if bc._is_dirichlet:
                phi = bc.clamp_dirichlet(phi)

        # Inner JIT-compiled solve.  residual_fn, boundary_conditions,
        # and extra args are captured by closure (compile-time constants);
        # only `phi` is a traced dynamic argument.
        # @jax.jit
        def _solve(phi):
            def cond_fn(state):
                _, converged, n_iter = state
                return jnp.logical_and(~converged, n_iter < max_iter)

            def body_fn(state):
                phi, _, n_iter = state
                for bc in boundary_conditions:
                    if bc._is_dirichlet:
                        phi = bc.clamp_dirichlet(phi)
                res = residual_fn(phi, boundary_conditions, *args)
                jac = jax.jacobian(residual_fn)(phi, boundary_conditions, *args)
                step = jnp.linalg.solve(jac, -res)
                phi = phi + step.reshape(phi.shape)
                step_norm = jnp.mean(jnp.square(step))
                converged = step_norm < tol
                return (phi, converged, n_iter + 1)

            init_state = (phi, jnp.bool_(False), jnp.int32(0))
            phi, converged, n_iter = jax.lax.while_loop(cond_fn, body_fn, init_state)
            # clamp once more after the final step update
            for bc in boundary_conditions:
                if bc._is_dirichlet:
                    phi = bc.clamp_dirichlet(phi)
            # Cut gradients: differentiating through the Newton loop is
            # not useful and would checkpoint every iteration's Jacobian.
            phi = jax.lax.stop_gradient(phi)
            residual = residual_fn(phi, boundary_conditions, *args)
            return phi, converged, n_iter, residual

        phi, converged, n_iter, residual = _solve(phi)
        return RootSolveResult(
            solution=phi,
            converged=converged,
            n_iter=n_iter,
            residual=residual,
        )
