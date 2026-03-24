# SPDX-License-Identifier: GPL-3.0-or-later
"""Nonlinear root-finding solvers for the FDM-EDL system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, ClassVar

import quaxed.numpy as jnp
import unxt

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

    solution_int: unxt.Quantity
    converged: bool
    n_iter: int
    residual: unxt.Quantity
    coordinate: unxt.Quantity = None
    solution: unxt.Quantity = None

    def set_coordinate(self, coordinate: unxt.Quantity) -> None:
        self.coordinate = coordinate

    def set_solution(self, solution: unxt.Quantity) -> None:
        self.solution = solution


class BaseSolver(ABC):
    """
    Abstract factory and base class for nonlinear root-finding solvers.

    Calling ``BaseSolver(method=...)`` returns a concrete solver instance
    selected from the registered sub-classes.

    Parameters
    ----------
    method : str, optional
        BaseSolver algorithm to instantiate (default: ``"newton"``).  Must
        match a key registered in :attr:`_registry`.

    Raises
    ------
    ValueError
        If *method* does not match any registered solver.
    """

    _registry: ClassVar[dict[str, type["BaseSolver"]]] = {}

    def __new__(cls, method: str = "newton", **kwargs):
        if cls is BaseSolver:
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
            BaseSolver._registry[method.lower()] = cls

    @abstractmethod
    def solve(
        self, residual_fn: ResidualFunction, phi0: unxt.Quantity, *args
    ) -> RootSolveResult:
        """
        Solve the nonlinear system ``residual_fn(phi, *args) = 0``.

        Parameters
        ----------
        residual_fn : callable
            Residual function with signature
            ``residual_fn(phi, *args) -> jax.Array``.
        phi0 : jax.Array
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
