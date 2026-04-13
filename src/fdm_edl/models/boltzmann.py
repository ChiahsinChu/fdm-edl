# SPDX-License-Identifier: GPL-3.0-or-later
"""Charge density models for the electrical double layer.

Each model provides a different approximation for the ionic charge
density as a function of the local electrostatic potential.
"""

from __future__ import annotations

from typing import Dict

# from typing import TYPE_CHECKING
import jax
import unxt
from jax import numpy as jnp

from ..api.electrolyte import Electrolyte
from ..utils import constants
from .base import ChargeModel, boltzmann_factor, register

# import quaxed.numpy as jnp
# import unxt


# ---------------------------------------------------------------------------
# Boltzmann (Gouy-Chapman)
# ---------------------------------------------------------------------------
@register("boltzmann")
class BoltzmannModel(ChargeModel):
    """Standard Boltzmann distribution (Gouy-Chapman theory).

    .. math::

        c_i(\\phi) = c_{i,\\infty} \\exp\\!\\left(-\\frac{z_i e \\phi}{k_B T}\\right)
    """

    def charge_density(
        self,
        phi: jax.Array,
        electrolyte: Electrolyte,
        temperature: float,
    ) -> jax.Array:
        # beta: (eV)^(-1)
        beta: float = 1.0 / (
            constants.BOLTZMANN_CONSTANT.to("eV / K").value * temperature
        )
        # rho_ion: e / angstrom^3
        rho_ion = jnp.zeros_like(phi)
        for ion in electrolyte.ions.values():
            z_i = ion._charge
            rho_ion += (
                z_i
                * ion._molar_conc
                * jnp.exp(-z_i * phi * beta)
                * constants.AVOGADRO_NUMBER.value
            )
        return rho_ion

    def ion_concentration_profile(
        self,
        phi: unxt.Quantity,
        electrolyte: Electrolyte,
        temperature: unxt.Quantity,
    ) -> Dict[str, unxt.Quantity]:
        return {
            name: ion.molar_conc * boltzmann_factor(phi, temperature, ion.charge)
            for name, ion in electrolyte.ions.items()
        }
