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
    """Abstract base class for solvent models used in the electrolyte."""

    _registry: ClassVar[dict[str, type["BaseSolvent"]]] = {}

    def __new__(cls, type: str = "uniform", **kwargs):
        if cls is BaseSolvent:
            method_key = str(type).lower()
            solvent_cls = cls._registry.get(method_key)
            if solvent_cls is None:
                supported = ", ".join(sorted(cls._registry))
                raise ValueError(
                    f"Unsupported solvent class '{type}'. Supported classes: {supported}."
                )
            # remove 'type' so ConstantEps.__init__ doesn't see it
            kwargs.pop("type", None)
            return solvent_cls(**kwargs)  # construct fully here
        return super().__new__(cls)

    def __init_subclass__(cls, *, types: tuple[str, ...] = (), **kwargs):
        super().__init_subclass__(**kwargs)
        for type in types:
            BaseSolvent._registry[type.lower()] = cls

    def compute_eps(self, efield: unxt.Quantity) -> jax.Array:
        """Compute the field-dependent relative permittivity for a given electric field.

        Parameters
        ----------
        efield : jax.Array
            Electric field in V/angstrom.

        Returns
        -------
        jax.Array
            Field-dependent relative permittivity.
        """
        x = efield.to(uc.UNIT_SYSTEMS["metal"]["electrical field strength"])
        return self._compute_eps(x.value)

    @abstractmethod
    def _compute_eps(self, efield: jax.Array) -> jax.Array:
        """Compute the field-dependent relative permittivity for a given electric field.

        Parameters
        ----------
        efield : jax.Array
            Electric field in V/angstrom.

        Returns
        -------
        jax.Array
            Field-dependent relative permittivity.
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


class UniformDielectrics(BaseSolvent, types=("uniform",)):
    """Constant dielectric response model with a fixed relative permittivity."""

    def __init__(self, epsilon_r: float = 78.5, **kwargs):
        self.epsilon_r = epsilon_r

    @property
    def eps_0(self) -> float:
        return self.epsilon_r

    @property
    def eps_inf(self) -> float:
        return self.epsilon_r

    def _compute_eps(self, efield: jax.Array) -> jax.Array:
        return jnp.full_like(efield, self.eps_0)
