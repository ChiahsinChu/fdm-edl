# SPDX-License-Identifier: GPL-3.0-or-later
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

Array = jax.Array


def _water_eps(coord: Array) -> Array:
    return jnp.full_like(coord, 78.4)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BaseGradientOP(ABC):
    uniform: bool = True
    boundary_points: int = 4
    interior_points: int = 3
    eps_func: Callable[[Array], Array] = _water_eps

    def __post_init__(self) -> None:
        if self.boundary_points not in (3, 4, 5):
            raise ValueError("boundary_points must be 3, 4, or 5.")
        if self.interior_points not in (3, 5):
            raise ValueError("interior_points must be 3 or 5.")
        if self.interior_points > self.boundary_points:
            raise ValueError(
                "interior_points should be <= boundary_points "
                "(increase boundary_points or reduce interior_points)."
            )

    # --- PyTree protocol (static aux data) ---
    def tree_flatten(self):
        children = ()
        aux_data = (
            self.uniform,
            self.boundary_points,
            self.interior_points,
            self.eps_func,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data: tuple[bool, int, int, Callable[[Array], Array] | None, str],
        children: tuple[()],
    ) -> "BaseGradientOP":
        uniform, boundary_points, interior_points, eps_func = aux_data
        return cls(
            uniform=uniform,
            boundary_points=boundary_points,
            interior_points=interior_points,
            eps_func=eps_func,
        )

    # --- public API ---
    def __call__(self, x: Array, y: Array) -> tuple[Array, Array]:
        """
        Compute the gradient and subclass-specific divergence term.

        Parameters
        ----------
        x : jax.numpy.ndarray
            Grid locations with shape ``(n,)``.
        y : jax.numpy.ndarray
            Function values with shape ``(n,)`` corresponding to ``y ≈ f(x)``.

        Returns
        -------
        grad : jax.numpy.ndarray
            First derivative approximation with shape ``(n,)``.
        div_D : jax.numpy.ndarray
            Subclass-defined divergence-like term with shape ``(n,)``.
        """
        sorted_idx = jnp.argsort(x)
        _x = x[sorted_idx]
        _y = y[sorted_idx]
        grad_phi = self._grad(_x, _y)
        div_D = self._div_D(_x, _y, grad_phi)
        return (grad_phi, div_D)

    @abstractmethod
    def _grad(self, x: Array, y: Array) -> Array:
        """
        Compute the gradient (first derivative) only.

        Parameters
        ----------
        x : jax.numpy.ndarray
            Grid locations with shape ``(n,)``.
        y : jax.numpy.ndarray
            Function values with shape ``(n,)``.

        Returns
        -------
        grad : jax.numpy.ndarray
            First derivative approximation with shape ``(n,)``.
        """
        raise NotImplementedError("Subclasses must implement _grad.")

    @abstractmethod
    def _div_D(self, x: Array, y: Array, grad: Array) -> Array:
        """
        Compute subclass-specific divergence-like term.

        Parameters
        ----------
        x : jax.numpy.ndarray
            Grid locations with shape ``(n,)``.
        y : jax.numpy.ndarray
            Function values with shape ``(n,)``.
        grad : jax.numpy.ndarray
            Gradient computed by ``_grad``, with shape ``(n,)``.

        Returns
        -------
        div_D : jax.numpy.ndarray
            Divergence-like term with shape ``(n,)``.
        """
        raise NotImplementedError("Subclasses must implement _div_D.")
