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
    alpha : float, optional
        Initial backtracking line-search step size (default: 1.0).
    max_iter : int, optional
        Maximum number of Newton iterations (default: 20).
    max_iter_ls : int, optional
        Maximum number of backtracking line-search steps (default: 10).
    atol_var : float or None, optional
        Absolute tolerance for step norm convergence (default: 1e-6).
    rtol_var : float or None, optional
        Relative tolerance for step norm convergence (default: None, ignored).
    atol_grad : float or None, optional
        Absolute tolerance for gradient norm convergence (default: None, ignored).
    rtol_grad : float or None, optional
        Relative tolerance for gradient norm convergence (default: None, ignored).
    atol_src : float or None, optional
        Absolute tolerance for source term convergence (default: None, ignored).
    rtol_src : float or None, optional
        Relative tolerance for source term convergence (default: None, ignored).
    atol_res : float or None, optional
        Absolute tolerance for residual convergence (default: None, ignored).
    rtol_res : float or None, optional
        Relative tolerance for residual convergence (default: None, ignored).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        max_iter: int = 20,
        max_iter_ls: int = 10,
        atol_var: float = 1e-6,
        rtol_var: float | None = None,
        atol_grad: float | None = None,
        rtol_grad: float | None = None,
        atol_src: float | None = None,
        rtol_src: float | None = None,
        atol_res: float | None = None,
        rtol_res: float | None = None,
        **kwargs,
    ):
        super().__init__(
            max_iter=max_iter,
            atol_var=atol_var,
            rtol_var=rtol_var,
            atol_grad=atol_grad,
            rtol_grad=rtol_grad,
            atol_src=atol_src,
            rtol_src=rtol_src,
            atol_res=atol_res,
            rtol_res=rtol_res,
        )
        self.alpha = alpha
        self.max_iter_ls = max_iter_ls

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
            ``residual_fn(phi, *args) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array, jax.Array]]``.
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
        # clamp Dirichlet nodes in the initial guess
        phi_init = self._clamp_dirichlet_nodes(phi, boundary_conditions)
        res_init, _aux = residual_fn(phi_init, boundary_conditions, *args)
        grad_init, _div_D, src_init = _aux

        # Inner JIT-compiled solve.  residual_fn, boundary_conditions,
        # and extra args are captured by closure (compile-time constants);
        # only `phi` is a traced dynamic argument.
        @jax.jit
        def _solve(phi):
            def cond_fn(state):
                return self._cond_fn(state, self.max_iter)

            def body_fn(state):
                phi_old, grad_old, src_old, res_old, _, n_iter = state

                jac, _aux = jax.jacobian(residual_fn, has_aux=True)(
                    phi_old, boundary_conditions, *args
                )
                step = jnp.linalg.solve(jac, -res_old).reshape(phi_old.shape)

                # Backtracking line search to reduce divergence/non-finite updates.
                def ls_body(_ii, ls_state):
                    alpha, best_phi, best_norm, accepted = ls_state
                    cand_phi = self._clamp_dirichlet_nodes(
                        phi + alpha * step, boundary_conditions
                    )
                    cand_res, _aux = residual_fn(cand_phi, boundary_conditions, *args)
                    cand_norm = self._residual_norm(cand_res)
                    cand_finite = jnp.logical_and(
                        jnp.all(jnp.isfinite(cand_phi)),
                        jnp.all(jnp.isfinite(cand_res)),
                    )
                    improved = jnp.logical_and(cand_finite, cand_norm < best_norm)
                    accepted = jnp.logical_or(accepted, improved)
                    best_phi = jnp.where(improved, cand_phi, best_phi)
                    best_norm = jnp.where(improved, cand_norm, best_norm)
                    return (0.5 * alpha, best_phi, best_norm, accepted)

                # alpha, best_phi, best_norm, accepted
                ls_init = (
                    jnp.asarray(self.alpha, dtype=phi.dtype),
                    phi_old,
                    self._residual_norm(res_old),
                    jnp.bool_(False),
                )
                _, ls_phi, _ls_norm, accepted = jax.lax.fori_loop(
                    0, self.max_iter_ls, ls_body, ls_init
                )

                full_phi = self._clamp_dirichlet_nodes(
                    phi_old + step, boundary_conditions
                )
                phi_new = jnp.where(accepted, ls_phi, full_phi)
                res_new, _aux = residual_fn(phi_new, boundary_conditions, *args)
                grad_new, _div_D, src_new = _aux

                # check convergence on step, gradient, source
                if self.atol_var is not None or self.rtol_var is not None:
                    atol = self.atol_var if self.atol_var is not None else 1e-6
                    rtol = self.rtol_var if self.rtol_var is not None else 1e-3
                    converged_step = jnp.allclose(
                        phi_new, phi_old, atol=atol, rtol=rtol
                    )
                else:
                    converged_step = jnp.bool_(True)
                if self.atol_grad is not None or self.rtol_grad is not None:
                    atol = self.atol_grad if self.atol_grad is not None else 1e-8
                    rtol = self.rtol_grad if self.rtol_grad is not None else 1e-5
                    converged_grad = jnp.allclose(
                        grad_new, grad_old, atol=atol, rtol=rtol
                    )
                else:
                    converged_grad = jnp.bool_(True)
                if self.atol_src is not None or self.rtol_src is not None:
                    atol = self.atol_src if self.atol_src is not None else 1e-8
                    rtol = self.rtol_src if self.rtol_src is not None else 1e-5
                    converged_src = jnp.allclose(src_new, src_old, atol=atol, rtol=rtol)
                else:
                    converged_src = jnp.bool_(True)
                # check the residual
                if self.atol_res is not None or self.rtol_res is not None:
                    atol = self.atol_res if self.atol_res is not None else 1e-8
                    rtol = self.rtol_res if self.rtol_res is not None else 1e-5
                    converged_res = jnp.allclose(
                        res_new, jnp.zeros_like(res_new), atol=atol, rtol=rtol
                    )
                else:
                    converged_res = jnp.bool_(True)
                converged = (
                    converged_step * converged_grad * converged_src * converged_res
                )
                return (phi_new, grad_new, src_new, res_new, converged, n_iter + 1)

            init_state = (
                phi_init,
                grad_init,
                src_init,
                res_init,
                jnp.bool_(False),
                jnp.int32(0),
            )
            phi_final, grad_final, src_final, res_final, converged, n_iter = (
                jax.lax.while_loop(cond_fn, body_fn, init_state)
            )
            # clamp once more after the final step update
            phi_final = self._clamp_dirichlet_nodes(phi_final, boundary_conditions)
            # Cut gradients: differentiating through the Newton loop is
            # not useful and would checkpoint every iteration's Jacobian.
            phi_final = jax.lax.stop_gradient(phi_final)
            return phi_final, grad_final, src_final, res_final, converged, n_iter

        phi, grad, src, res, converged, n_iter = _solve(phi)
        return RootSolveResult(
            solution=phi,
            converged=converged,
            n_iter=n_iter,
            residual=res,
            gradient=grad,
            source=src,
        )
