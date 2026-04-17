# SPDX-License-Identifier: GPL-3.0-or-later
"""Nonlinear root-finding solvers for the FDM-EDL system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from jax import numpy as jnp

if TYPE_CHECKING:
    from typing import ClassVar, Sequence

    import jax

from ..utils.bc import BoundaryCondition

ResidualFunction = Callable[..., jnp.ndarray]


@dataclass(frozen=True)
class RootSolveResult:
    """
    Container for the output of a nonlinear root-finding solve.

    Attributes
    ----------
    solution : jax.Array
        Converged (or best-effort) solution vector from the solver.
    converged : bool
        ``True`` if the solver met its convergence criterion.
    n_iter : int
        Number of iterations performed.
    residual : jax.Array
        Residual vector evaluated at ``solution``.
    """

    solution: jax.Array
    converged: bool
    n_iter: int
    residual: jax.Array

    def __post_init__(self):
        assert self.n_iter >= 0, "n_iter must be non-negative"
        assert (
            self.solution.shape == self.residual.shape
        ), "solution and residual must have the same shape"


class BaseSolver(ABC):
    """
    Abstract factory and base class for nonlinear root-finding solvers.

    Calling ``BaseSolver(method=...)`` returns a concrete solver instance
    selected from the registered sub-classes.  The *method* keyword
    (default: ``"newton"``) must match a key in the internal registry.

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
        self,
        residual_fn: ResidualFunction,
        phi: jax.Array,
        boundary_conditions: Sequence[BoundaryCondition],
        *args,
    ) -> RootSolveResult:
        """
        Solve the nonlinear system ``residual_fn(phi, *args) = 0``.

        Parameters
        ----------
        residual_fn : callable
            Residual function with signature
            ``residual_fn(phi, *args) -> jax.Array``.
        phi : jax.Array
            Initial guess for the solution.
        boundary_conditions : Sequence[BoundaryCondition]
            List of boundary conditions to apply during the solve.
        *args
            Extra positional arguments forwarded to *residual_fn*.

        Returns
        -------
        RootSolveResult
            Result object containing the solution, convergence flag,
            iteration count, and final residual norm.
        """
        raise NotImplementedError
