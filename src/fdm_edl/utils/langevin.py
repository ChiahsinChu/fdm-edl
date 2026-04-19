# SPDX-License-Identifier: GPL-3.0-or-later
import jax
import unxt
from astropy.units import cds  # type: ignore[import-untyped]
from jax import numpy as jnp

from . import constants


def langevin_function(x: jax.Array) -> jax.Array:
    """Evaluate the Langevin function.

    Parameters
    ----------
    x : jax.Array
        Dimensionless field-strength argument.

    Returns
    -------
    jax.Array
        Value of the Langevin function, using a small-argument expansion near
        zero for numerical stability.
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
    """Compute the field-dependent dielectric contribution from dipolar alignment.

    Parameters
    ----------
    efield : jax.Array
        Electric field magnitude in V/angstrom.
    temperature : float
        Temperature in K.
    mu : float
        Molecular dipole moment in e*angstrom.
    n_density : float
        Number density in 1/angstrom^3.

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


def booth_eps(
    efield: jax.Array,
    temperature: float,
    eps_0: float,
    eps_opt: float,
    booth_beta: float,
) -> jax.Array:
    """Compute the field-dependent dielectric contribution from dipolar alignment.

    Parameters
    ----------
    efield : jax.Array
        Electric field magnitude in V/angstrom.
    temperature : float
        Temperature in K.
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


class LangevinWaterEps:
    """Field-dependent dielectric response model for liquid water.

    Parameters
    ----------
    temperature : unxt.Quantity
        Absolute temperature of the water phase.
    """

    def __init__(self, temperature: unxt.Quantity):
        self.temperature = temperature

        # optical dielectric constant of water at room temperature
        self.eps_opt = 1.78

        # dipole moment of water molecule in gas phase, converted to e * angstrom
        self.mu = unxt.Quantity(3.0, "debye").to(cds.e * unxt.unit("angstrom"))

        rho_water = unxt.Quantity(997, "kg/m^3")
        molar_mass_water = unxt.Quantity(18.01528, "g/mol")
        conc_water = (rho_water / molar_mass_water).to("mol/L")
        # number density of water molecules in 1/angstrom^3
        self.n_density = (conc_water * constants.AVOGADRO_NUMBER).to("1/angstrom^3")

    def __call__(self, efield: unxt.Quantity) -> jax.Array:
        """Evaluate the total relative permittivity for a physical electric field.

        Parameters
        ----------
        efield : unxt.Quantity
            Electric field in units convertible to V/angstrom.

        Returns
        -------
        jax.Array
            Relative permittivity including the optical contribution.
        """

        x = efield.to("V/angstrom").value
        return self.compute(x) + self.eps_opt

    def compute(self, efield: jax.Array) -> jax.Array:
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

        out = jax.vmap(langevin_eps, in_axes=(0, None, None, None))(
            _efield,
            self.temperature.value,
            self.mu.value,
            self.n_density.value,
        )
        return out.squeeze()

    @property
    def eps_static(self):
        """float: Static relative permittivity including the optical contribution."""

        return (
            (self.mu**2 * self.n_density)
            / (
                3
                * constants.BOLTZMANN_CONSTANT
                * self.temperature
                * constants.VACUUM_PERMITTIVITY
            )
        ).to("").value + self.eps_opt


class BoothWaterEps:
    """Field-dependent dielectric response model for liquid water based on Booth's theory.

    Parameters
    ----------
    temperature : unxt.Quantity
        Absolute temperature of the water phase.
    """

    def __init__(self, temperature: unxt.Quantity):
        self.temperature = temperature

        # optical dielectric constant of water at room temperature
        self.eps_opt = 1.78

        # static dielectric constant of water at room temperature
        self.eps_static = 78.4

        # Booth saturation parameter for water at room temperature in m/V, converted to angstrom/V
        self.booth_beta = unxt.Quantity(1.41e-8, "m/V").to("angstrom/V").value

    def __call__(self, efield: unxt.Quantity) -> jax.Array:
        """Evaluate the total relative permittivity for a physical electric field.

        Parameters
        ----------
        efield : unxt.Quantity
            Electric field in units convertible to V/angstrom.

        Returns
        -------
        jax.Array
            Relative permittivity including the optical contribution.
        """

        x = efield.to("V/angstrom").value
        return self.compute(x) + self.eps_opt

    def compute(self, efield: jax.Array) -> jax.Array:
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

        out = jax.vmap(booth_eps, in_axes=(0, None, None, None, None))(
            _efield,
            self.temperature.value,
            self.eps_static,
            self.eps_opt,
            self.booth_beta,
        )
        return out.squeeze()
