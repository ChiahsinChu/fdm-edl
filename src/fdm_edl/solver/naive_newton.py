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
            if bc.is_dirichlet:
                phi = bc.clamp_dirichlet(phi)

        # Inner JIT-compiled solve.  residual_fn, boundary_conditions,
        # and extra args are captured by closure (compile-time constants);
        # only `phi` is a traced dynamic argument.
        # @jax.jit
        def _solve(phi):
            def _clamp_dirichlet_nodes(phi_vec):
                for bc in boundary_conditions:
                    if bc.is_dirichlet:
                        phi_vec = bc.clamp_dirichlet(phi_vec)
                return phi_vec

            def _residual_norm(res):
                return jnp.mean(jnp.square(res))

            def cond_fn(state):
                _, converged, n_iter = state
                return jnp.logical_and(~converged, n_iter < max_iter)

            def body_fn(state):
                phi, _, n_iter = state
                phi = _clamp_dirichlet_nodes(phi)
                res = residual_fn(phi, boundary_conditions, *args)
                res_norm = _residual_norm(res)
                jac = jax.jacobian(residual_fn)(phi, boundary_conditions, *args)
                step = jnp.linalg.solve(jac, -res).reshape(phi.shape)

                # Backtracking line search to reduce divergence/non-finite updates.
                def ls_body(_, ls_state):
                    alpha, best_phi, best_norm, accepted = ls_state
                    cand_phi = _clamp_dirichlet_nodes(phi + alpha * step)
                    cand_res = residual_fn(cand_phi, boundary_conditions, *args)
                    cand_norm = _residual_norm(cand_res)
                    cand_finite = jnp.logical_and(
                        jnp.all(jnp.isfinite(cand_phi)),
                        jnp.all(jnp.isfinite(cand_res)),
                    )
                    improved = jnp.logical_and(cand_finite, cand_norm < best_norm)
                    best_phi = jnp.where(improved, cand_phi, best_phi)
                    best_norm = jnp.where(improved, cand_norm, best_norm)
                    accepted = jnp.logical_or(accepted, improved)
                    return (alpha * 0.5, best_phi, best_norm, accepted)

                ls_init = (
                    jnp.asarray(1.0, dtype=phi.dtype),
                    phi,
                    res_norm,
                    jnp.bool_(False),
                )
                _, ls_phi, ls_norm, accepted = jax.lax.fori_loop(0, 8, ls_body, ls_init)

                full_phi = _clamp_dirichlet_nodes(phi + step)
                phi_next = jnp.where(accepted, ls_phi, full_phi)
                step_norm = jnp.mean(jnp.square(phi_next - phi))
                converged = step_norm < tol
                return (phi_next, converged, n_iter + 1)

            init_state = (phi, jnp.bool_(False), jnp.int32(0))
            phi, converged, n_iter = jax.lax.while_loop(cond_fn, body_fn, init_state)
            # clamp once more after the final step update
            phi = _clamp_dirichlet_nodes(phi)
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
