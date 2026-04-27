# SPDX-License-Identifier: GPL-3.0-or-later
from abc import ABC, abstractmethod
from typing import ClassVar

import jax
import unxt
from jax import numpy as jnp

from ...utils import constants
from ...utils import unit_conversion as uc

_EPS_0 = constants.VACUUM_PERMITTIVITY.to(
    uc.UNIT_SYSTEMS["metal"]["permittivity"]
).value


class BaseSolvent(ABC):
    """Abstract base class for solvent dielectric response models.

    Subclasses provide the relative permittivity :math:`\epsilon_r(E)` for a
    given electric-field magnitude.
    """

    _registry: ClassVar[dict[str, type["BaseSolvent"]]] = {}
    type: ClassVar[str | tuple[str, ...] | None] = None

    def __new__(cls, type: str = "uniform", **kwargs):
        if cls is BaseSolvent:
            method_key = str(type).lower()
            solvent_cls = cls._registry.get(method_key)
            if solvent_cls is None:
                supported = ", ".join(sorted(cls._registry))
                raise ValueError(
                    f"Unsupported solvent class '{type}'. Supported classes: {supported}."
                )
            kwargs.pop("type", None)
            return solvent_cls(**kwargs)  # construct fully here
        return super().__new__(cls)

    def __init_subclass__(cls, *, types: tuple[str, ...] = (), **kwargs):
        super().__init_subclass__(**kwargs)
        class_type = getattr(cls, "type", None)

        if class_type is None:
            normalized_types = types
        elif isinstance(class_type, str):
            normalized_types = (class_type,)
        else:
            normalized_types = tuple(class_type)

        for type in normalized_types:
            BaseSolvent._registry[type.lower()] = cls

    def compute_eps(self, efield: unxt.Quantity) -> jax.Array:
        """Compute relative permittivity from a unit-bearing electric field.

        Parameters
        ----------
        efield : unxt.Quantity
            Electric field quantity. Values are converted to the internal
            ``metal`` unit system before evaluation.

        Returns
        -------
        jax.Array
            Relative permittivity ``epsilon_r`` (dimensionless).
        """
        x = efield.to(uc.UNIT_SYSTEMS["metal"]["electrical field strength"])
        return self._compute_eps(jnp.abs(x.value))

    @abstractmethod
    def _compute_eps(self, efield: jax.Array) -> jax.Array:
        """Compute relative permittivity from raw internal-unit field values.

        Parameters
        ----------
        efield : jax.Array
            Electric field magnitude in the internal ``metal`` unit system
            (numerical array, unitless at this stage).

        Returns
        -------
        jax.Array
            Relative permittivity ``epsilon_r`` (dimensionless).
        """
        ...

    @property
    @abstractmethod
    def eps_0(self) -> float:
        """Return the static relative permittivity."""
        ...

    @property
    @abstractmethod
    def eps_inf(self) -> float:
        """Return the optical/high-frequency relative permittivity."""
        ...

    @property
    def _eps_0(self) -> float:
        """Absolute static permittivity in the metal unit system."""

        return self.eps_0 * _EPS_0

    @property
    def _eps_inf(self) -> float:
        """Absolute optical permittivity in the metal unit system."""
        return self.eps_inf * _EPS_0


class UniformDielectrics(BaseSolvent):
    """Constant dielectric response model with a fixed relative permittivity."""

    type = "uniform"

    def __init__(self, epsilon_r: float = 78.5, **kwargs):
        """Initialize a uniform-dielectric model.

        Parameters
        ----------
        epsilon_r : float, default=78.5
            Constant relative permittivity applied for all field values.
        """
        self.epsilon_r = epsilon_r

    @property
    def eps_0(self) -> float:
        return self.epsilon_r

    @property
    def eps_inf(self) -> float:
        return self.epsilon_r

    def _compute_eps(self, efield: jax.Array) -> jax.Array:
        return jnp.full_like(efield, self.eps_0)
