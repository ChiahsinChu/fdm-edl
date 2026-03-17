# SPDX-License-Identifier: GPL-3.0-or-later
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
    x: jnp.ndarray
    converged: bool
    n_iter: int
    residual_norm: float


class Solver(ABC):
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
        raise NotImplementedError


class NewtonSolver(Solver, methods=("newton",)):
    def __init__(
        self, method: str | None = None, max_iter: int = 15, tol: float = 1e-6
    ):
        self.max_iter = max_iter
        self.tol = tol

    def solve(
        self, residual_fn: ResidualFunction, x0: jnp.ndarray, *args
    ) -> RootSolveResult:
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
            x=x,
            converged=converged,
            n_iter=n_iter,
            residual_norm=residual_norm,
        )


class JaxoptBroydenSolver(Solver, methods=("jaxopt_broyden", "broyden")):
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
            x=params,
            converged=converged,
            n_iter=n_iter,
            residual_norm=residual_norm,
        )
