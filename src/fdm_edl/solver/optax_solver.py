# SPDX-License-Identifier: GPL-3.0-or-later
"""Nonlinear root-finding solvers for the FDM-EDL system."""

from __future__ import annotations

from typing import Literal, Sequence

import jax
import optax
import quaxed.numpy as jnp
import unxt

from fdm_edl.bc import BoundaryCondition
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
    bc_enforcement : ``"hard"`` or ``"penalty"``, optional
        How boundary conditions are enforced (default: ``"hard"``).

        * ``"hard"`` – BCs overwrite the residual at boundary nodes
          (original behaviour).
        * ``"penalty"`` – BCs are enforced via a PINN-style penalty
          term added to the loss.  The physics residual is kept at all
          nodes (including boundary nodes) and the total loss becomes
          ``L_physics + penalty_weight * L_BC``.
    penalty_weight : float, optional
        Weight λ for the BC penalty term (default: 1.0).  Only used
        when ``bc_enforcement="penalty"``.

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
        bc_enforcement: Literal["hard", "penalty"] = "hard",
        penalty_weight: float = 1.0,
    ):
        self.max_iter = max_iter
        self.tol_step = tol_step
        self.tol_residual = tol_residual
        self.learning_rate = learning_rate
        self.bc_enforcement = bc_enforcement
        self.penalty_weight = penalty_weight

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
        self,
        residual_fn: ResidualFunction,
        phi0: unxt.Quantity,
        *args,
        boundary_conditions: Sequence[BoundaryCondition] | None = None,
        coordinates: unxt.Quantity | None = None,
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
        boundary_conditions : sequence of BoundaryCondition or None
            Required when ``bc_enforcement="penalty"``.  The BC objects
            whose :meth:`compute_violation` methods will supply the
            penalty loss terms.
        coordinates : unxt.Quantity or None
            Grid coordinates passed to :meth:`compute_violation`.
            Required when ``bc_enforcement="penalty"``.

        Returns
        -------
        RootSolveResult
            Result containing the solution and solver diagnostics.
        """
        phi_int = phi0
        opt_state = self.optimizer.init(phi_int)
        converged = False

        if self.bc_enforcement == "penalty":
            if boundary_conditions is None or coordinates is None:
                raise ValueError(
                    "boundary_conditions and coordinates are required "
                    "when bc_enforcement='penalty'."
                )
            # Flatten coordinates for 1-D BCs
            coords_for_bc = (
                coordinates[:, 0]
                if coordinates.ndim == 2 and coordinates.shape[1] == 1
                else coordinates
            )
            lam = self.penalty_weight

            def loss_fn(phi: unxt.Quantity) -> jax.Array:
                res = residual_fn(phi, *args)
                physics_loss = 0.5 * jnp.mean(res * res)
                bc_loss = jnp.zeros_like(physics_loss)
                for bc in boundary_conditions:
                    v = bc.compute_violation(phi, coords_for_bc)
                    bc_loss = bc_loss + 0.5 * jnp.mean(v * v)
                return physics_loss + lam * bc_loss
        else:

            def loss_fn(phi: unxt.Quantity) -> jax.Array:
                res = residual_fn(phi, *args)
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
