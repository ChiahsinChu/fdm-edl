# SPDX-License-Identifier: GPL-3.0-or-later
"""Charge density models for the electrical double layer.

Each model provides a different approximation for the ionic charge
density as a function of the local electrostatic potential.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import unxt
from jax import numpy as jnp

from ..utils import constants
from .base import ChargeModel, boltzmann_factor, register

if TYPE_CHECKING:
    from typing import Dict

    import jax

    from ..api.electrolyte import Electrolyte


# ---------------------------------------------------------------------------
# Bikerman (lattice-gas / steric)
# ---------------------------------------------------------------------------
@register("bikerman")
class BikermanModel(ChargeModel):
    r"""Generalized Bikerman (lattice-gas) model with per-ion radii.

    .. math::

        c_i(\phi) = \frac{c_{i,\infty}\,\exp(-z_i e\phi / k_BT)}
                         {1 + \sum_j \nu_j\,[\exp(-z_j e\phi / k_BT) - 1]}

    where :math:`\nu_j = N_A\,(2 r_j)^3\,c_{j,\infty}` is the bulk
    packing fraction of ion *j*.  Reduces to the Boltzmann model when
    all radii are zero.
    """

    def charge_density(
        self,
        phi: jax.Array,
        electrolyte: Electrolyte,
        temperature: float,
    ) -> jax.Array:
        """Calculate ionic charge density according to Bikerman model.

        Computes the charge density including steric effects from finite
        ion sizes using the generalized Bikerman lattice-gas model.

        Parameters
        ----------
        phi : jax.Array
            Electrostatic potential (eV).
        electrolyte : Electrolyte
            Electrolyte object containing ion properties and concentrations.
        temperature : float
            Temperature (K).

        Returns
        -------
        jax.Array
            Ionic charge density (e / angstrom^3).

        Notes
        -----
        The Bikerman model accounts for the finite volume occupied by ions,
        reducing to the Boltzmann model when all ion radii are zero.
        """
        # beta: (eV)^(-1)
        beta: float = 1.0 / (
            constants.BOLTZMANN_CONSTANT.to("eV / K").value * temperature
        )
        # rho_ion: e / angstrom^3
        rho_ion = jnp.zeros_like(phi)
        sfactor = 1.0
        for ion in electrolyte.ions.values():
            z_i = ion._charge
            rho_ion += (
                z_i
                * constants.AVOGADRO_NUMBER.value
                * ion._molar_conc
                * jnp.exp(-z_i * phi * beta)
            )
            sfactor += (
                ion._molar_conc * ion._molar_volume * (jnp.exp(-z_i * phi * beta) - 1.0)
            )
        return rho_ion / sfactor

    def ion_concentration_profile(
        self,
        phi: unxt.Quantity,
        electrolyte: Electrolyte,
        temperature: unxt.Quantity,
    ) -> Dict[str, unxt.Quantity]:
        """Calculate ion concentration profile with steric effects.

        Computes the local ion concentrations at each point in space using
        the Bikerman model, which includes both electrostatic and steric effects.

        Parameters
        ----------
        phi : unxt.Quantity
            Electrostatic potential with units.
        electrolyte : Electrolyte
            Electrolyte object containing ion properties and concentrations.
        temperature : unxt.Quantity
            Temperature with units.

        Returns
        -------
        Dict[str, unxt.Quantity]
            Dictionary mapping ion names to their concentration profiles (with units).
            Each concentration field includes the effects of the local potential
            and finite ion sizes.

        Notes
        -----
        The steric factor accounts for the volume fraction of ions at the
        equilibrium packing limit, preventing infinite concentrations at high
        potentials.
        """
        boltzmann_conc = {}
        sfactor = 1.0
        for name, ion in electrolyte.ions.items():
            boltzmann_conc[name] = ion.molar_conc * boltzmann_factor(
                phi, temperature, ion.charge
            )
            sfactor += (ion.molar_conc * ion.molar_volume).to("") * (
                boltzmann_factor(phi, temperature, ion.charge) - 1.0
            )
        return {name: conc / sfactor for name, conc in boltzmann_conc.items()}
