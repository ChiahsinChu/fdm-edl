# SPDX-License-Identifier: GPL-3.0-or-later
import jax
import unxt
from astropy.units import cds  # type: ignore[import-untyped]
from jax import numpy as jnp

from ...api.edl import ElectricalDoubleLayer
from ...utils import constants
from ...utils import unit_conversion as uc
from .base import BaseSolvent


def langevin_function(x: jax.Array) -> jax.Array:
    """Evaluate the Langevin function.

    Parameters
    ----------
    x : jax.Array
        Dimensionless field-strength argument.

    Returns
    -------
    jax.Array
        Value of the Langevin function, using a low-order series expansion
        near zero for numerical stability.
    """

    def small(_):
        return x / 3 - x**3 / 45

    def large(_):
        return 1.0 / jnp.tanh(x) - 1.0 / x

    return jax.lax.cond(x < 1e-10, small, large, operand=None)


def langevin_eps(
    efield: jax.Array,
    temperature: float,
    mu: float,
    n_density: float,
) -> jax.Array:
    """Compute Langevin orientational dielectric contribution.

    Parameters
    ----------
    efield : jax.Array
        Electric field magnitude in the internal ``metal`` field unit.
    temperature : float
        Temperature in Kelvin.
    mu : float
        Molecular dipole moment in internal ``metal`` dipole units.
    n_density : float
        Number density in internal ``metal`` density units.

    Returns
    -------
    jax.Array
        Dimensionless dielectric contribution.
    """

    kB = constants.BOLTZMANN_CONSTANT.to("eV/K").value
    eps0 = constants.VACUUM_PERMITTIVITY.to(cds.e / unxt.unit("V * angstrom")).value
    coeff_in = mu / (kB * temperature)
    coeff_out = (mu * n_density) / eps0

    def _func(E):
        return coeff_out / E * langevin_function(coeff_in * E)

    def small_field(_):
        # constant in E => grad wrt E is exactly 0
        return coeff_in * coeff_out / 3

    def large_field(_):
        return _func(efield)

    return jax.lax.cond(efield > 1e-3, large_field, small_field, operand=None)


class LangevinDielectrics(BaseSolvent, types=("langevin",)):
    """Field-dependent dielectric response model based on Langevin dipole alignment."""

    def __init__(
        self,
        edl_obj: ElectricalDoubleLayer,
        epsilon_r_inf: float = 1.78,
        mu: float = None,
        # rho: float = None,
        # molar_mass: float = None,
        **kwargs,
    ):
        """Initialize the Langevin dielectric model.

        Parameters
        ----------
        edl_obj : ElectricalDoubleLayer
            EDL object providing temperature and unit system.
        epsilon_r_inf : float, default=1.78
            Optical/high-frequency relative permittivity.
        mu : float | None, default=None
            Molecular dipole moment. If ``None``, a water-like default of
            ``3.0 debye`` is used.
        """
        self._epsilon_r_inf = epsilon_r_inf

        self.temperature = edl_obj.temperature
        # dipole moment of water molecule in gas phase, converted to e * angstrom
        self.mu = (
            unxt.Quantity(
                mu,
                unit=uc.UNIT_SYSTEMS[edl_obj._unit_system]["electrical dipole moment"],
            ).to(uc.UNIT_SYSTEMS["metal"]["electrical dipole moment"])
            if mu is not None
            else unxt.Quantity(3.0, "debye").to(
                uc.UNIT_SYSTEMS["metal"]["electrical dipole moment"]
            )
        )

        # todo: take rho and molar mass as input and convert to number density
        rho = unxt.Quantity(997, "kg/m^3")
        molar_mass = unxt.Quantity(18.01528, "g/mol")
        molar_conc = (rho / molar_mass).to("mol/L")
        # number density
        self.n_density = (molar_conc * constants.AVOGADRO_NUMBER).to(
            unxt.unit("") / uc.UNIT_SYSTEMS["metal"]["volume"]
        )

    @property
    def eps_0(self) -> float:
        return (
            (self.mu**2 * self.n_density)
            / (
                3
                * constants.BOLTZMANN_CONSTANT
                * self.temperature
                * constants.VACUUM_PERMITTIVITY
            )
        ).to("").value + self._epsilon_r_inf

    @property
    def eps_inf(self) -> float:
        return self._epsilon_r_inf

    def _compute_eps(self, efield: jax.Array) -> jax.Array:
        """Evaluate Langevin dielectric response for raw internal-unit fields.

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

        out = jax.vmap(langevin_eps, in_axes=(0, None, None, None))(
            _efield,
            self.temperature.value,
            self.mu.value,
            self.n_density.value,
        )
        return out.squeeze() + self.eps_inf


class LangevinWater(LangevinDielectrics):
    """Preconfigured Langevin model for liquid water-like response."""

    def __init__(self, edl_obj: ElectricalDoubleLayer):
        super().__init__(
            edl_obj=edl_obj,
            epsilon_r_inf=1.78,
        )
