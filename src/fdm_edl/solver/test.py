# SPDX-License-Identifier: GPL-3.0-or-later
"""Nonlinear root-finding solvers for the FDM-EDL system."""

from __future__ import annotations

import jax
import quaxed.numpy as jnp
import unxt

from fdm_edl.solver.base import BaseSolver, ResidualFunction, RootSolveResult


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
        self, method: str | None = None, max_iter: int = 15, tol: float = 1e-6
    ):
        self.max_iter = max_iter
        self.tol = tol

    def solve(
        self, residual_fn: ResidualFunction, phi0: unxt.Quantity, *args
    ) -> RootSolveResult:
        """
        Solve via Newton-Raphson iterations with a full JAX Jacobian.

        Parameters
        ----------
        residual_fn : callable
            Residual function with signature
            ``residual_fn(phi, *args) -> jax.Array``.
        phi0 : jax.Array
            Initial guess.
        *args
            Extra positional arguments forwarded to *residual_fn*.

        Returns
        -------
        RootSolveResult
            Result containing the solution and solver diagnostics.
        """
        phi_int = phi0
        converged = False

        for ii in range(self.max_iter):
            res = residual_fn(phi_int, *args)
            _jac = jax.jacfwd(residual_fn)(phi_int, *args)
            jac = unxt.Quantity(_jac.value.value, unit=_jac.unit / phi_int.unit)
            step = jnp.linalg.solve(jac, -res)
            phi_int = phi_int + step.reshape(phi_int.shape)

            # check convergence based on step norm relative to number of variables
            step_norm = jnp.mean(step.value * step.value)
            if step_norm < self.tol:
                converged = True
                break

        residual = residual_fn(phi_int, *args)
        return RootSolveResult(
            solution_int=phi_int,
            converged=converged,
            n_iter=ii + 1,
            residual=residual,
        )


# class JaxoptBroydenSolver(BaseSolver, methods=("jaxopt_broyden", "broyden")):
#     """
#     Broyden quasi-Newton solver backed by *jaxopt*.

#     Parameters
#     ----------
#     method : str or None, optional
#         Ignored; present for API compatibility with the factory.
#     max_iter : int, optional
#         Maximum number of Broyden iterations (default: 100).
#     tol : float, optional
#         Convergence tolerance (default: 1e-6).

#     Attributes
#     ----------
#     max_iter : int
#         Maximum iteration count.
#     tol : float
#         Convergence threshold.

#     Raises
#     ------
#     ImportError
#         If *jaxopt* is not installed.
#     """

#     def __init__(
#         self, method: str | None = None, max_iter: int = 100, tol: float = 1e-6
#     ):
#         self.max_iter = max_iter
#         self.tol = tol
#         try:
#             Broyden = importlib.import_module("jaxopt").Broyden
#         except ImportError as exc:
#             raise ImportError(
#                 "jaxopt is not installed. Install it with `pip install jaxopt` to use JaxoptBroydenSolver."
#             ) from exc
#         self._broyden_cls = Broyden

#     def solve(
#         self, residual_fn: ResidualFunction, phi0: unxt.Quantity, *args
#     ) -> RootSolveResult:
#         """
#         Solve via Broyden's quasi-Newton method using *jaxopt*.

#         Parameters
#         ----------
#         residual_fn : callable
#             Residual function with signature
#             ``residual_fn(phi, *args) -> jax.Array``.
#         phi0 : jax.Array
#             Initial guess.
#         *args
#             Extra positional arguments forwarded to *residual_fn*.

#         Returns
#         -------
#         RootSolveResult
#             Result containing the solution and solver diagnostics.
#         """
#         solver = self._broyden_cls(
#             optimality_fun=residual_fn,
#             maxiter=self.max_iter,
#             tol=self.tol,
#             implicit_diff=False,
#         )
#         result = solver.run(phi0, *args)
#         params = result.params
#         state = result.state
#         residual_norm = float(jnp.linalg.norm(residual_fn(params, *args)))
#         converged = bool(getattr(state, "error", jnp.inf) < self.tol)
#         n_iter = int(getattr(state, "iter_num", self.max_iter))
#         return RootSolveResult(
#             solution=params,
#             converged=converged,
#             n_iter=n_iter,
#             residual_norm=residual_norm,
#         )
