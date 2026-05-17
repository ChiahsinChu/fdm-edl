# SPDX-License-Identifier: GPL-3.0-or-later
"""Nonlinear root-finding solvers for the FDM-EDL system."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
from jax import numpy as jnp

if TYPE_CHECKING:
    from typing import Sequence

from jax.scipy.sparse.linalg import bicgstab, cg, gmres

from ..utils.bc import BoundaryCondition
from .base import BaseSolver, ResidualFunction, RootSolveResult


class ScipySolver(BaseSolver):
    def __init__(
        self,
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
        self.max_iter_ls = max_iter_ls

        kwargs.pop("method", None)  # ignore method if passed in kwargs
        self.kwargs = kwargs

        self.scipy_solver = None

    def solve(
        self,
        residual_fn: ResidualFunction,
        phi: jax.Array,
        boundary_conditions: Sequence[BoundaryCondition],
        *args,
    ) -> RootSolveResult:
        """
        Solve with Newton iterations and a matrix-free Krylov linear solver.

        Each outer Newton step linearises the residual and solves the
        resulting system with the configured iterative solver
        (``bicgstab``, ``cg``, or ``gmres``). The solve requires only
        Jacobian-vector products computed by ``jax.jvp`` (no explicit
        Jacobian is formed). The outer loop uses
        ``jax.lax.while_loop`` so the entire solve is JIT-compilable,
        enabling early termination on convergence.

        Parameters
        ----------
        residual_fn : callable
            Residual function with signature
            ``residual_fn(phi, *args) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array, jax.Array]]``
            returning ``(residual, (gradient, div_D, source))``.
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
        def _solve(phi_init):
            def cond_fn(state):
                return self._cond_fn(state, self.max_iter)

            def body_fn(state):
                phi_old, grad_old, src_old, res_old, _, n_iter = state

                # Build a residual function that traces only phi.
                # boundary_conditions and *args stay in closure as static Python data.
                def residual_phi_only(phi_var):
                    res_var, _ = residual_fn(phi_var, boundary_conditions, *args)
                    return res_var

                # Matrix-free Jacobian-vector product J(phi_old) @ v.
                def jvp_fn(vec):
                    _, jvp = jax.jvp(residual_phi_only, (phi_old,), (vec,))
                    return jvp

                # Newton step equation: Jacobian @ delta_u = -F(u)
                # jvp_fn takes a vector and returns (Jacobian @ vector)
                step, _info = self.scipy_solver(
                    jvp_fn, -res_old, maxiter=self.max_iter_ls, **self.kwargs
                )

                # Update our solution guess
                phi_new = self._clamp_dirichlet_nodes(
                    phi_old + step, boundary_conditions
                )
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

        phi, grad, src, res, converged, n_iter = _solve(phi_init)
        return RootSolveResult(
            solution=phi,
            converged=converged,
            n_iter=n_iter,
            residual=res,
            gradient=grad,
            source=src,
        )


class BiCGStabSolver(ScipySolver, methods=("bicgstab",)):
    """
    BIConjugate Gradient STABilized (BiCGStab) solver.

    Uses a matrix-free Jacobian-vector product (JVP) via ``jax.jvp`` to
    avoid forming the full Jacobian.  The outer loop iterates Newton
    steps; each step solves the linearised system with
    ``jax.scipy.sparse.linalg.bicgstab``.

    Parameters
    ----------
    alpha : float, optional
        Unused step-size parameter kept for API compatibility (default: 1.0).
    max_iter : int, optional
        Maximum number of outer Newton iterations (default: 20).
    max_iter_ls : int, optional
        Maximum number of inner BiCGStab iterations per Newton step
        (default: 10).
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
            max_iter_ls=max_iter_ls,
            atol_var=atol_var,
            rtol_var=rtol_var,
            atol_grad=atol_grad,
            rtol_grad=rtol_grad,
            atol_src=atol_src,
            rtol_src=rtol_src,
            atol_res=atol_res,
            rtol_res=rtol_res,
            **kwargs,
        )

        self.scipy_solver = bicgstab


class CGSolver(ScipySolver, methods=("cg",)):
    """
    Conjugate Gradient (CG) solver.

    Uses a matrix-free Jacobian-vector product (JVP) via ``jax.jvp`` to
    avoid forming the full Jacobian.  The outer loop iterates Newton
    steps; each step solves the linearised system with
    ``jax.scipy.sparse.linalg.cg``.

    Parameters
    ----------
    max_iter : int, optional
        Maximum number of outer Newton iterations (default: 20).
    max_iter_ls : int, optional
        Maximum number of inner CG iterations per Newton step
        (default: 10).
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
            max_iter_ls=max_iter_ls,
            atol_var=atol_var,
            rtol_var=rtol_var,
            atol_grad=atol_grad,
            rtol_grad=rtol_grad,
            atol_src=atol_src,
            rtol_src=rtol_src,
            atol_res=atol_res,
            rtol_res=rtol_res,
            **kwargs,
        )

        self.scipy_solver = cg


class GMRESSolver(ScipySolver, methods=("gmres",)):
    """Generalized Minimal Residual (GMRES) solver.

    Uses a matrix-free Jacobian-vector product (JVP) via ``jax.jvp`` to
    avoid forming the full Jacobian. The outer loop iterates Newton
    steps; each step solves the linearised system with
    ``jax.scipy.sparse.linalg.gmres``.

    Parameters
    ----------
    max_iter : int, optional
        Maximum number of outer Newton iterations (default: 20).
    max_iter_ls : int, optional
        Maximum number of inner GMRES iterations per Newton step
        (default: 10).
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
            max_iter_ls=max_iter_ls,
            atol_var=atol_var,
            rtol_var=rtol_var,
            atol_grad=atol_grad,
            rtol_grad=rtol_grad,
            atol_src=atol_src,
            rtol_src=rtol_src,
            atol_res=atol_res,
            rtol_res=rtol_res,
            **kwargs,
        )

        self.scipy_solver = gmres
