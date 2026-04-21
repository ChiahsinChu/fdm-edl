# SPDX-License-Identifier: GPL-3.0-or-later

import jax
import unxt

from ...api.edl import ElectricalDoubleLayer
from ...utils import unit_conversion as uc
from .base import BaseSolvent
from .langevin import langevin_function


def booth_eps(
    efield: jax.Array,
    eps_0: float,
    eps_opt: float,
    booth_beta: float,
) -> jax.Array:
    """Compute the field-dependent dielectric contribution from dipolar alignment.

    Parameters
    ----------
    efield : jax.Array
        Electric field magnitude in V/angstrom.
    eps_0 : float
        Static dielectric constant.
    eps_opt : float
        Optical dielectric constant.
    booth_beta : float
        Booth saturation parameter (1.41e-8 m/V for water at room temperature).

    Returns
    -------
    jax.Array
        Dimensionless dielectric contribution.
    """
    coeff_in = 3 * (eps_0 - eps_opt) / booth_beta
    coeff_out = booth_beta

    def _func(E):
        return coeff_out / E * langevin_function(coeff_in * E)

    def small_field(_):
        # constant in E => grad wrt E is exactly 0
        return coeff_in * coeff_out / 3

    def large_field(_):
        return _func(efield)

    return jax.lax.cond(efield > 1e-2, large_field, small_field, operand=None)


class BoothDielectrics(BaseSolvent, types=("langevin",)):
    """Field-dependent dielectric response model based on Langevin dipole alignment."""

    def __init__(
        self,
        edl_obj: ElectricalDoubleLayer,
        epsilon_r_0: float = 78.4,
        epsilon_r_inf: float = 1.78,
        booth_beta: float = None,
        **kwargs,
    ):
        self._epsilon_r_0 = epsilon_r_0
        self._epsilon_r_inf = epsilon_r_inf

        # Booth saturation parameter for water at room temperature in m/V
        self.booth_beta = (
            unxt.Quantity(
                booth_beta,
                unit=uc.UNIT_SYSTEMS[edl_obj._unit_system]["length"]
                / uc.UNIT_SYSTEMS[edl_obj._unit_system]["electrical potential"],
            )
            if booth_beta is not None
            else unxt.Quantity(
                1.41e-8,
                "m/V",
            )
        )
        self._booth_beta: float = self.booth_beta.to(
            uc.UNIT_SYSTEMS["metal"]["length"]
            / uc.UNIT_SYSTEMS["metal"]["electrical potential"]
        ).value

    @property
    def eps_0(self) -> float:
        return self._epsilon_r_0

    @property
    def eps_inf(self) -> float:
        return self._epsilon_r_inf

    def _compute_eps(self, efield: jax.Array) -> jax.Array:
        """Evaluate the orientational dielectric contribution for raw field values.

        Parameters
        ----------
        efield : jax.Array
            Electric field in V/angstrom.

        Returns
        -------
        jax.Array
            Field-dependent orientational dielectric contribution.
        """

        _efield = efield[None] if (efield.ndim == 0) else efield

        out = jax.vmap(booth_eps, in_axes=(0, None, None, None))(
            _efield,
            self._epsilon_r_0,
            self._epsilon_r_inf,
            self._booth_beta,
        )
        return out.squeeze() + self.eps_inf


class BoothWater(BoothDielectrics):
    """Field-dependent dielectric response model for liquid water based on Langevin dipole alignment."""

    def __init__(self, edl_obj: ElectricalDoubleLayer):
        super().__init__(
            edl_obj=edl_obj,
            epsilon_r_inf=1.78,
            epsilon_r_0=78.4,
        )
