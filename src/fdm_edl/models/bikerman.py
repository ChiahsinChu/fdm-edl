# SPDX-License-Identifier: GPL-3.0-or-later
"""Charge density models for the electrical double layer.

Each model provides a different approximation for the ionic charge
density as a function of the local electrostatic potential.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import unxt
from jax import numpy as jnp
from jax.scipy.special import logsumexp

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
        beta = 1.0 / (constants.BOLTZMANN_CONSTANT.to("eV / K").value * temperature)

        ions = list(electrolyte.ions.values())
        z = jnp.asarray([ion._charge for ion in ions])  # (n_ions,)
        c = jnp.asarray([ion._molar_conc for ion in ions])  # (n_ions,)
        v = jnp.asarray([ion._molar_volume for ion in ions])  # (n_ions,)

        # x: (n_ions, *phi.shape)
        x = -z[:, None] * phi[None, ...] * beta
        a = c * v  # (n_ions,)

        # ---- numerator: sum_i (NA * c_i * z_i * exp(x_i)) ----
        # We must handle sign because z_i can be negative.
        w = constants.AVOGADRO_NUMBER.value * c * z  # (n_ions,)
        w_abs = jnp.abs(w)
        w_sign = jnp.sign(w)

        # log(|w_i| * exp(x_i)) = log|w_i| + x_i
        # Use -inf for zero weights to avoid nan in log.
        log_w_abs = jnp.where(w_abs > 0, jnp.log(w_abs), -jnp.inf)[:, None]
        t_num = log_w_abs + x

        # signed sum: sum_i sign_i * exp(t_num_i)
        tmax = jnp.max(t_num, axis=0)
        num = jnp.exp(tmax) * jnp.sum(w_sign[:, None] * jnp.exp(t_num - tmax), axis=0)

        # ---- denominator: s = 1 - sum(a) + sum_i (a_i * exp(x_i)) ----
        # Compute Sexp = sum_i a_i * exp(x_i) stably in log-space:
        log_a = jnp.where(a > 0, jnp.log(a), -jnp.inf)[:, None]
        log_Sexp = logsumexp(log_a + x, axis=0)  # log(sum_i a_i exp(x_i))
        Sexp = jnp.exp(log_Sexp)

        a_sum = jnp.sum(a)
        s = (1.0 - a_sum) + Sexp

        # Optional safety: if numerical noise makes s <= 0, clamp.
        # Physically s should be > 0 for valid parameters.
        s = jnp.maximum(s, jnp.finfo(phi.dtype).tiny)

        return num / s

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
        # Ensure sfactor stays positive to avoid NaN from division
        # Clamp to small positive value when steric effects cause sfactor to approach zero
        if isinstance(sfactor, unxt.Quantity):
            sfactor = unxt.Quantity(
                jnp.maximum(sfactor.value, jnp.finfo(sfactor.value.dtype).tiny),
                sfactor.unit,
            )
        else:
            sfactor = jnp.maximum(jnp.asarray(sfactor), 1e-10)
        return {name: conc / sfactor for name, conc in boltzmann_conc.items()}
