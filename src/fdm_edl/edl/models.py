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

from .. import _constants

if TYPE_CHECKING:
    from .electrolyte import Electrolyte, Ion
    from .system import boltzmann_factor


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_MODEL_REGISTRY: dict[str, type[ChargeModel]] = {}


def _register(name: str):
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
        phi: unxt.Quantity,
        electrolyte: Electrolyte,
        temperature: unxt.Quantity,
    ) -> unxt.Quantity:
        """Total ionic charge density ρ_ion [C/m³] at every grid node.

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
        unxt.Quantity, shape (n_grid,)
        """
        ...

    @abstractmethod
    def ion_concentration(
        self,
        phi: unxt.Quantity,
        ion: Ion,
        temperature: unxt.Quantity,
    ) -> unxt.Quantity:
        """Local molar concentration of a single ionic species.

        Parameters
        ----------
        phi : unxt.Quantity, shape (n_grid,)
            Electrostatic potential.
        ion : Ion
            Ionic species.
        temperature : unxt.Quantity
            Absolute temperature.

        Returns
        -------
        unxt.Quantity, shape (n_grid,)
            Concentration in the same unit as ``ion.molar_conc``.
        """
        ...


# ---------------------------------------------------------------------------
# Boltzmann (Gouy-Chapman)
# ---------------------------------------------------------------------------
@_register("boltzmann")
class BoltzmannModel(ChargeModel):
    """Standard Boltzmann distribution (Gouy-Chapman theory).

    .. math::

        c_i(\\phi) = c_{i,\\infty} \\exp\\!\\left(-\\frac{z_i e \\phi}{k_B T}\\right)
    """

    def charge_density(self, phi, electrolyte, temperature):
        phi_V = phi.to("V")
        beta = 1.0 / (_constants.BOLTZMANN_CONSTANT * temperature).to("eV")
        rho_ion = unxt.Quantity(0.0, "C / m^3")
        for ion in electrolyte.ions:
            charge = ion.charge.to("C")
            exponent = (-charge * phi_V * beta).to("")
            rho_ion += (
                charge * ion.molar_conc * _constants.AVOGADRO_NUMBER * jnp.exp(exponent)
            )
        return rho_ion

    def ion_concentration(self, phi, ion, temperature):
        valency = (ion.charge.to("C") / _constants.ELEMENTARY_CHARGE).to("").value
        return ion.molar_conc * boltzmann_factor(phi, temperature, int(valency))


# ---------------------------------------------------------------------------
# Bikerman (lattice-gas / steric)
# ---------------------------------------------------------------------------
@_register("bikerman")
class BikermanModel(ChargeModel):
    r"""Generalized Bikerman (lattice-gas) model with per-ion radii.

    .. math::

        c_i(\phi) = \frac{c_{i,\infty}\,\exp(-z_i e\phi / k_BT)}
                         {1 + \sum_j \nu_j\,[\exp(-z_j e\phi / k_BT) - 1]}

    where :math:`\nu_j = N_A\,(2 r_j)^3\,c_{j,\infty}` is the bulk
    packing fraction of ion *j*.  Reduces to the Boltzmann model when
    all radii are zero.
    """

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _packing_fractions(electrolyte: Electrolyte) -> list[float]:
        """Compute ν_j for each ion."""
        nu = []
        for ion in electrolyte.ions:
            r = ion.radius.to("m").value
            c = ion.molar_conc.to("mol/m^3").value
            N_A = _constants.AVOGADRO_NUMBER.to("1/mol").value
            nu.append(N_A * (2.0 * r) ** 3 * c)
        return nu

    @staticmethod
    def _boltzmann_exponents(phi, electrolyte, temperature):
        """Return list of exp(-z_i e φ / k_B T) arrays."""
        phi_V = phi.to("V")
        kBT = _constants.BOLTZMANN_CONSTANT * temperature
        exps = []
        for ion in electrolyte.ions:
            charge = ion.charge.to("C")
            exponent = (-charge * phi_V / kBT).to("")
            exps.append(jnp.exp(exponent))
        return exps

    def _denominator(self, phi, electrolyte, temperature):
        """1 + Σ_j ν_j [exp(-z_j e φ / k_BT) - 1]."""
        nus = self._packing_fractions(electrolyte)
        boltz = self._boltzmann_exponents(phi, electrolyte, temperature)
        denom = 1.0
        for nu_j, bf_j in zip(nus, boltz):
            denom = denom + nu_j * (bf_j - 1.0)
        return denom

    # -- interface --------------------------------------------------------
    def charge_density(self, phi, electrolyte, temperature):
        denom = self._denominator(phi, electrolyte, temperature)
        boltz = self._boltzmann_exponents(phi, electrolyte, temperature)

        rho_ion = unxt.Quantity(0.0, "C / m^3")
        for ion, bf in zip(electrolyte.ions, boltz):
            charge = ion.charge.to("C")
            bulk_concentration = ion.molar_conc.to("mol/m^3")
            rho_ion += (
                charge * bulk_concentration * _constants.AVOGADRO_NUMBER * bf / denom
            )
        return rho_ion

    def ion_concentration(self, phi, ion, temperature):
        # For a single ion we still need the full denominator, so we
        # require the caller to use the system-level helper that passes
        # the full electrolyte.  Here we provide a partial that works when
        # the denominator has been pre-computed and stashed.
        raise NotImplementedError(
            "Use ElectricalDoubleLayer.get_ion_concentration_profiles() "
            "for Bikerman — it passes the full electrolyte to compute "
            "the denominator."
        )

    def ion_concentration_full(
        self,
        phi: unxt.Quantity,
        electrolyte: Electrolyte,
        temperature: unxt.Quantity,
    ) -> dict[str, unxt.Quantity]:
        """Compute all ion concentration profiles at once.

        Returns
        -------
        dict[str, unxt.Quantity]
            Mapping of ion name → local molar concentration.
        """
        denom = self._denominator(phi, electrolyte, temperature)
        boltz = self._boltzmann_exponents(phi, electrolyte, temperature)

        profiles: dict[str, unxt.Quantity] = {}
        for ion, bf in zip(electrolyte.ions, boltz):
            profiles[ion.name] = ion.molar_conc * bf / denom
        return profiles
