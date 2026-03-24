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
    Root-finding via minimization of 1/2 ||residual||^2 using Optax.

    Notes
    -----
    We minimize: loss(phi) = 0.5 * sum(residual(phi)^2)
    and stop based on step norm (like NewtonSolver) and/or residual norm.
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
