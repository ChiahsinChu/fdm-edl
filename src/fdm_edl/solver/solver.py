# SPDX-License-Identifier: GPL-3.0-or-later
"""Nonlinear root-finding solvers for the FDM-EDL system."""
from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, ClassVar

import jax
import jax.numpy as jnp

ResidualFunction = Callable[..., jnp.ndarray]


@dataclass
class RootSolveResult:
    """
    Container for the output of a nonlinear root-finding solve.

    Attributes
    ----------
    solution : jax.Array
        Converged (or best-effort) solution vector.
    converged : bool
        ``True`` if the solver met its convergence criterion.
    n_iter : int
        Number of iterations performed.
    residual_norm : float
        Euclidean norm of the residual evaluated at ``solution``.
    """

    solution: jnp.ndarray
    converged: bool
    n_iter: int
    residual_norm: float


class Solver(ABC):
    """
    Abstract factory and base class for nonlinear root-finding solvers.

    Calling ``Solver(method=...)`` returns a concrete solver instance
    selected from the registered sub-classes.

    Parameters
    ----------
    method : str, optional
        Solver algorithm to instantiate (default: ``"newton"``).  Must
        match a key registered in :attr:`_registry`.

    Raises
    ------
    ValueError
        If *method* does not match any registered solver.
    """

    _registry: ClassVar[dict[str, type["Solver"]]] = {}

    def __new__(cls, method: str = "newton", **kwargs):
        if cls is Solver:
            method_key = str(method).lower()
            solver_cls = cls._registry.get(method_key)
            if solver_cls is None:
                supported = ", ".join(sorted(cls._registry))
                raise ValueError(
                    f"Unsupported solver method '{method}'. Supported methods: {supported}."
                )
            return super().__new__(solver_cls)
        return super().__new__(cls)

    def __init_subclass__(cls, *, methods: tuple[str, ...] = (), **kwargs):
        super().__init_subclass__(**kwargs)
        for method in methods:
            Solver._registry[method.lower()] = cls

    @abstractmethod
    def solve(
        self, residual_fn: ResidualFunction, x0: jnp.ndarray, *args
    ) -> RootSolveResult:
        """
        Solve the nonlinear system ``residual_fn(x, *args) = 0``.

        Parameters
        ----------
        residual_fn : callable
            Residual function with signature
            ``residual_fn(x, *args) -> jax.Array``.
        x0 : jax.Array
            Initial guess for the solution.
        *args
            Extra positional arguments forwarded to *residual_fn*.

        Returns
        -------
        RootSolveResult
            Result object containing the solution, convergence flag,
            iteration count, and final residual norm.
        """
        raise NotImplementedError


class NewtonSolver(Solver, methods=("newton",)):
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
        self, residual_fn: ResidualFunction, x0: jnp.ndarray, *args
    ) -> RootSolveResult:
        """
        Solve via Newton-Raphson iterations with a full JAX Jacobian.

        Parameters
        ----------
        residual_fn : callable
            Residual function with signature
            ``residual_fn(x, *args) -> jax.Array``.
        x0 : jax.Array
            Initial guess.
        *args
            Extra positional arguments forwarded to *residual_fn*.

        Returns
        -------
        RootSolveResult
            Result containing the solution and solver diagnostics.
        """
        x = x0
        converged = False
        n_iter = 0

        for i in range(self.max_iter):
            res = residual_fn(x, *args)
            jac = jax.jacobian(residual_fn)(x, *args)
            step = jnp.linalg.solve(jac, -res)
            x = x + step
            step_norm = float(jnp.linalg.norm(step))
            n_iter = i + 1
            if step_norm < self.tol:
                converged = True
                break

        residual_norm = float(jnp.linalg.norm(residual_fn(x, *args)))
        return RootSolveResult(
            solution=x,
            converged=converged,
            n_iter=n_iter,
            residual_norm=residual_norm,
        )


class JaxoptBroydenSolver(Solver, methods=("jaxopt_broyden", "broyden")):
    """
    Broyden quasi-Newton solver backed by *jaxopt*.

    Parameters
    ----------
    method : str or None, optional
        Ignored; present for API compatibility with the factory.
    max_iter : int, optional
        Maximum number of Broyden iterations (default: 100).
    tol : float, optional
        Convergence tolerance (default: 1e-6).

    Attributes
    ----------
    max_iter : int
        Maximum iteration count.
    tol : float
        Convergence threshold.

    Raises
    ------
    ImportError
        If *jaxopt* is not installed.
    """

    def __init__(
        self, method: str | None = None, max_iter: int = 100, tol: float = 1e-6
    ):
        self.max_iter = max_iter
        self.tol = tol
        try:
            Broyden = importlib.import_module("jaxopt").Broyden
        except ImportError as exc:
            raise ImportError(
                "jaxopt is not installed. Install it with `pip install jaxopt` to use JaxoptBroydenSolver."
            ) from exc
        self._broyden_cls = Broyden

    def solve(
        self, residual_fn: ResidualFunction, x0: jnp.ndarray, *args
    ) -> RootSolveResult:
        """
        Solve via Broyden's quasi-Newton method using *jaxopt*.

        Parameters
        ----------
        residual_fn : callable
            Residual function with signature
            ``residual_fn(x, *args) -> jax.Array``.
        x0 : jax.Array
            Initial guess.
        *args
            Extra positional arguments forwarded to *residual_fn*.

        Returns
        -------
        RootSolveResult
            Result containing the solution and solver diagnostics.
        """
        solver = self._broyden_cls(
            optimality_fun=residual_fn,
            maxiter=self.max_iter,
            tol=self.tol,
            implicit_diff=False,
        )
        result = solver.run(x0, *args)
        params = result.params
        state = result.state
        residual_norm = float(jnp.linalg.norm(residual_fn(params, *args)))
        converged = bool(getattr(state, "error", jnp.inf) < self.tol)
        n_iter = int(getattr(state, "iter_num", self.max_iter))
        return RootSolveResult(
            solution=params,
            converged=converged,
            n_iter=n_iter,
            residual_norm=residual_norm,
        )
