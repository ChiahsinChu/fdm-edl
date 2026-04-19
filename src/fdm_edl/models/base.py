# SPDX-License-Identifier: GPL-3.0-or-later
"""Charge density models for the electrical double layer.

Each model provides a different approximation for the ionic charge
density as a function of the local electrostatic potential.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import quaxed.numpy as jnp
import unxt

from ..api.electrolyte import Electrolyte

if TYPE_CHECKING:
    from typing import Dict
    import jax
from ..utils import constants
from ..utils.unit_conversion import UNIT_SYSTEMS

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_MODEL_REGISTRY: dict[str, type[ChargeModel]] = {}


def register(name: str):
    """Class decorator that registers a ChargeModel subclass."""

    def wrapper(cls):
        _MODEL_REGISTRY[name] = cls
        return cls

    return wrapper


def create_charge_model(params: dict) -> ChargeModel:
    """Instantiate a :class:`ChargeModel` from a parameter dictionary.

    Parameters
    ----------
    params : dict
        Must contain ``"type"`` (str) matching a registered model name.
        Remaining keys are forwarded to the constructor.

    Returns
    -------
    ChargeModel
    """
    params = copy.deepcopy(params)  # deep copy
    model_type = params.pop("type", "boltzmann")
    cls = _MODEL_REGISTRY.get(model_type)
    if cls is None:
        raise ValueError(
            f"Unknown charge model '{model_type}'. "
            f"Available: {list(_MODEL_REGISTRY)}"
        )
    return cls(**params)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
class ChargeModel(ABC):
    """Abstract base class for ionic charge density models."""

    @abstractmethod
    def charge_density(
        self,
        phi: jax.Array,
        electrolyte: Electrolyte,
        temperature: float,
    ) -> jax.Array:
        """Total ionic charge density ρ_ion [C/m³] at every grid node.

        Parameters
        ----------
        phi : jax.Array, shape (n_grid,)
            Electrostatic potential in the internal unit system (V).
        electrolyte : Electrolyte
            Electrolyte definition (ions, permittivity, …).
        temperature : float
            Absolute temperature in Kelvin.

        Returns
        -------
        jax.Array, shape (n_grid,)
            Ionic charge density in the internal unit system (e/angstrom^3).
        """
        ...

    @abstractmethod
    def ion_concentration_profile(
        self,
        phi: unxt.Quantity,
        electrolyte: Electrolyte,
        temperature: unxt.Quantity,
    ) -> Dict[str, unxt.Quantity]:
        """Local molar concentration of a single ionic species.

        Parameters
        ----------
        phi : unxt.Quantity, shape (n_grid,)
            Electrostatic potential.
        electrolyte : Electrolyte
            Electrolyte definition (ions, permittivity, …).
        temperature : unxt.Quantity
            Absolute temperature.

        Returns
        -------
        Dict[str, unxt.Quantity], shape (n_grid,)
            Concentrations for each ionic species in the same units as their respective ``molar_conc``.
        """
        ...


# @jax.jit
def boltzmann_factor(
    phi: unxt.Quantity,
    temperature: unxt.Quantity,
    charge: unxt.Quantity,
):
    """Compute the Boltzmann factor exp(-z e φ / k_B T).

    Parameters
    ----------
    phi : unxt.Quantity
        Electrostatic potential.
    temperature : unxt.Quantity
        Absolute temperature.
    charge : unxt.Quantity
        Ion charge.

    Returns
    -------
    unxt.Quantity
        Dimensionless Boltzmann factor at each grid node.
    """
    beta = 1.0 / (constants.BOLTZMANN_CONSTANT * temperature)
    return jnp.exp((-charge * phi * beta).to(""))


def charge_density_profile(
    ion_concentration_profile: Dict[str, unxt.Quantity],
) -> unxt.Quantity:
    """Compute the ionic charge density profile from ion concentration profiles."""
    if len(ion_concentration_profile) == 0:
        raise ValueError("ion_concentration_profile must not be empty")

    ion_conc_iter = iter(ion_concentration_profile.values())
    ion_conc = next(ion_conc_iter)
    for _conc in ion_conc_iter:
        ion_conc = ion_conc + _conc
    ion_conc = ion_conc.to("mol / L")
    # from molar concentration (mol/L) to charge density (e/Å³)
    rho_ion = (ion_conc * constants.AVOGADRO_NUMBER * constants.ELEMENTARY_CHARGE).to(
        UNIT_SYSTEMS["metal"]["electrical charge density"]
    )
    return rho_ion
