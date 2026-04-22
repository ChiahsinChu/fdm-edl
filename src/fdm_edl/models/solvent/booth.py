# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import unxt

from ...utils import unit_conversion as uc
from .base import BaseSolvent
from .langevin import langevin_function

if TYPE_CHECKING:
    from ...api.edl import ElectricalDoubleLayer


def booth_eps(
    efield: jax.Array,
    eps_0: float,
    eps_opt: float,
    booth_beta: float,
) -> jax.Array:
    """Compute Booth orientational dielectric contribution.

    Parameters
    ----------
    efield : jax.Array
        Electric field magnitude in the internal ``metal`` field unit.
    eps_0 : float
        Static relative permittivity.
    eps_opt : float
        Optical/high-frequency relative permittivity.
    booth_beta : float
        Booth saturation parameter in inverse field units.

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


class BoothDielectrics(BaseSolvent):
    """Field-dependent dielectric response model based on Langevin dipole alignment."""

    type = "booth"

    def __init__(
        self,
        edl_obj: ElectricalDoubleLayer,
        epsilon_r_0: float = 78.4,
        epsilon_r_inf: float = 1.78,
        booth_beta: float = None,
        **kwargs,
    ):
        """Initialize the Booth dielectric model.

        Parameters
        ----------
        edl_obj : ElectricalDoubleLayer
            EDL object used to interpret user-supplied units.
        epsilon_r_0 : float, default=78.4
            Static relative permittivity.
        epsilon_r_inf : float, default=1.78
            Optical/high-frequency relative permittivity.
        booth_beta : float | None, default=None
            Booth saturation parameter. If ``None``, uses ``1.41e-8 m/V``.
        """
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
        """Evaluate Booth dielectric response for raw internal-unit fields.

        Parameters
        ----------
        efield : jax.Array
            Electric field magnitude in internal ``metal`` units.

        Returns
        -------
        jax.Array
            Relative permittivity including ``eps_inf``.
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
    """Preconfigured Booth model for liquid water at room conditions."""

    def __init__(self, edl_obj: ElectricalDoubleLayer):
        super().__init__(
            edl_obj=edl_obj,
            epsilon_r_inf=1.78,
            epsilon_r_0=78.4,
        )
