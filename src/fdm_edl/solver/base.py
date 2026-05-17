# SPDX-License-Identifier: GPL-3.0-or-later
"""Nonlinear root-finding solvers for the FDM-EDL system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Tuple

from jax import numpy as jnp

if TYPE_CHECKING:
    from typing import ClassVar, Sequence

    import jax

from ..utils.bc import BoundaryCondition

ResidualFunction = Callable[
    ..., Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]
]


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
    gradient : jax.Array or None
        Electric-field vector (gradient of the potential) at ``solution``,
        or ``None`` if not returned by the residual function.
    source : jax.Array or None
        Source term evaluated at ``solution``, or ``None`` if not returned
        by the residual function.
    """

    solution: jax.Array
    converged: bool
    n_iter: int
    residual: jax.Array
    gradient: jax.Array | None = None
    source: jax.Array | None = None

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

    def __init__(
        self,
        max_iter: int = 20,
        atol_var: float = 1e-6,
        rtol_var: float | None = None,
        atol_grad: float | None = None,
        rtol_grad: float | None = None,
        atol_src: float | None = None,
        rtol_src: float | None = None,
        atol_res: float | None = None,
        rtol_res: float | None = None,
    ):
        self.max_iter = max_iter
        self.atol_var = atol_var
        self.rtol_var = rtol_var
        self.atol_grad = atol_grad
        self.rtol_grad = rtol_grad
        self.atol_src = atol_src
        self.rtol_src = rtol_src
        self.atol_res = atol_res
        self.rtol_res = rtol_res

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
            ``residual_fn(phi, *args) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array, jax.Array]]``
            returning ``(residual, (gradient, div_D, source))``.
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
            iteration count, residual, gradient, and source term.
        """
        raise NotImplementedError

    @staticmethod
    def _residual_norm(res: jax.Array) -> jax.Array:
        """Compute the scalar residual norm used for convergence checking."""
        return jnp.mean(jnp.square(res))

    @staticmethod
    def _clamp_dirichlet_nodes(
        phi: jax.Array,
        bcs: Sequence[BoundaryCondition],
    ) -> jax.Array:
        """Clamp Dirichlet nodes in the solution vector."""
        for bc in bcs:
            if bc.is_dirichlet:
                phi = bc.clamp_dirichlet(phi)
        return phi

    @staticmethod
    def _cond_fn(state, max_iter: int) -> jax.Array:
        _phi, _grad, _src, _res, converged, n_iter = state
        # continue iterating if not converged and under max iterations
        return jnp.logical_and(~converged, n_iter < max_iter)

    @staticmethod
    def _convergence_flag(a, b, atol: float, rtol: float) -> jax.Array:
        if atol is not None or rtol is not None:
            _atol = atol if atol is not None else 1e-8
            _rtol = rtol if rtol is not None else 1e-5
            return jnp.allclose(a, b, atol=_atol, rtol=_rtol)
        else:
            return jnp.bool_(True)
