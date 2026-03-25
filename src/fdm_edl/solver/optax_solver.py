# SPDX-License-Identifier: GPL-3.0-or-later
"""Nonlinear root-finding solvers for the FDM-EDL system."""

from __future__ import annotations

import jax
import optax
import quaxed.numpy as jnp
import unxt

from fdm_edl.solver.base import BaseSolver, ResidualFunction, RootSolveResult


class OptaxSolver(BaseSolver, methods=("optax", "adam", "sgd", "rmsprop")):
    """
    Root-finding via minimization of ½ ||residual||² using Optax.

    Parameters
    ----------
    method : str or None, optional
        Optax optimizer name: ``"adam"`` (default), ``"sgd"``, or
        ``"rmsprop"``.
    max_iter : int, optional
        Maximum number of optimization steps (default: 200).
    tol_step : float, optional
        Convergence tolerance on the mean step norm (default: 1e-6).
    tol_residual : float or None, optional
        If given, also require the residual norm to be below this
        threshold for convergence.
    learning_rate : float, optional
        Learning rate passed to the optimizer (default: 1e-2).
    optimizer : optax.GradientTransformation or None, optional
        Pre-built Optax optimizer.  When provided, *method* and
        *learning_rate* are ignored.

    Notes
    -----
    Minimises ``loss(φ) = 0.5 * Σ residual(φ)²`` and stops based on step
    norm and, optionally, residual norm.
    """

    def __init__(
        self,
        method: str | None = "adam",
        max_iter: int = 200,
        tol_step: float = 1e-6,
        tol_residual: float | None = None,
        learning_rate: float = 1e-2,
        optimizer: optax.GradientTransformation | None = None,
    ):
        self.max_iter = max_iter
        self.tol_step = tol_step
        self.tol_residual = tol_residual
        self.learning_rate = learning_rate

        # Choose an optimizer if one wasn't provided
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            m = (method or "adam").lower()
            if m in ("optax", "adam"):
                self.optimizer = optax.adam(learning_rate)
            elif m == "sgd":
                self.optimizer = optax.sgd(learning_rate)
            elif m == "rmsprop":
                self.optimizer = optax.rmsprop(learning_rate)
            else:
                raise ValueError(f"Unknown optax method '{method}'.")

    def solve(
        self, residual_fn: ResidualFunction, phi0: unxt.Quantity, *args
    ) -> RootSolveResult:
        """Solve by gradient-based minimization of the residual norm.

        Parameters
        ----------
        residual_fn : callable
            Residual function with signature
            ``residual_fn(phi, *args) -> unxt.Quantity``.
        phi0 : unxt.Quantity
            Initial guess for the solution.
        *args
            Extra positional arguments forwarded to *residual_fn*.

        Returns
        -------
        RootSolveResult
            Result containing the solution and solver diagnostics.
        """
        phi_int = phi0
        opt_state = self.optimizer.init(phi_int)
        converged = False

        def loss_fn(phi: unxt.Quantity) -> jax.Array:
            # residual is a Quantity (vector)
            res = residual_fn(phi, *args)
            # scalar, unit: (res.unit)^2
            return 0.5 * jnp.sum(res * res)

        # value_and_grad works with pytrees; should work with unxt.Quantity if it's a pytree.
        value_and_grad = jax.value_and_grad(loss_fn)

        for ii in range(self.max_iter):
            loss, grad = value_and_grad(phi_int)

            updates, opt_state = self.optimizer.update(grad, opt_state, params=phi_int)
            phi_next = optax.apply_updates(phi_int, updates)

            step = phi_next - phi_int
            phi_int = phi_next

            # Step norm convergence (dimensionless-ish, like your Newton solver)
            step_norm = jnp.linalg.norm(step).value / len(phi_int)

            # Optional residual-norm convergence
            if self.tol_residual is not None:
                res = residual_fn(phi_int, *args)
                res_norm = jnp.linalg.norm(res).value
                if (step_norm < self.tol_step) and (res_norm < self.tol_residual):
                    converged = True
                    break
            else:
                if step_norm < self.tol_step:
                    converged = True
                    break

        residual = residual_fn(phi_int, *args)
        return RootSolveResult(
            solution_int=phi_int,
            converged=converged,
            n_iter=ii + 1,
            residual=residual,
        )
