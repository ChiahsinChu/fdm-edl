# SPDX-License-Identifier: GPL-3.0-or-later
import quaxed.numpy as jnp
import unxt

from fdm_edl.edl import ElectricalDoubleLayer
from fdm_edl.edl.system import boltzmann_factor

from .. import _constants


class BasePoissonBoltzmann:
    """Shared utilities for analytical Poisson-Boltzmann models in 1D."""

    def __init__(
        self,
        edl_obj: ElectricalDoubleLayer,
        x: unxt.Quantity,
        sigma: unxt.Quantity,
    ):
        """Initialise the analytical PB model.

        Parameters
        ----------
        edl_obj : ElectricalDoubleLayer
            Configured EDL system (electrolyte, temperature, etc.).
        x : unxt.Quantity
            Spatial coordinates at which to evaluate the solution.
        sigma : unxt.Quantity
            Surface charge density.
        """
        self.edl_obj = edl_obj
        self.beta = 1.0 / (_constants.BOLTZMANN_CONSTANT * self.edl_obj.temperature).to(
            "eV"
        )

        self.x = x.to("nm")
        self.sigma = sigma.to("C/angstrom^2")

    @property
    def exp_factor(self) -> unxt.Quantity:
        """Dimensionless exponential decay factor exp(-x / λ_D)."""
        return jnp.exp(-(self.x / self.edl_obj.electrolyte.debye_length)).to("")


class LinearPoissonBoltzmann(BasePoissonBoltzmann):
    """Analytical linearized Poisson-Boltzmann profile in 1D.

    Users provide an ElectricalDoubleLayer object and can optionally provide
    custom spatial coordinates and surface charge density.
    """

    @property
    def phi(self) -> unxt.Quantity:
        """Linearised PB potential profile in Volts."""
        return (
            self.sigma
            * self.edl_obj.electrolyte.debye_length
            / self.edl_obj.electrolyte.epsilon
            * self.exp_factor
        ).to("V")

    @property
    def rho(self) -> unxt.Quantity:
        """Linearised PB charge density profile in mol/L."""
        _rho = -self.sigma / self.edl_obj.electrolyte.debye_length * self.exp_factor
        return (_rho / (_constants.ELEMENTARY_CHARGE * _constants.AVOGADRO_NUMBER)).to(
            "mol/L"
        )


class NonLinearPoissonBoltzmann(BasePoissonBoltzmann):
    """Analytical non-linear Poisson-Boltzmann profile in 1D.

    Users provide an ElectricalDoubleLayer object and can optionally provide
    custom spatial coordinates and surface charge density.

    Notes
    -----
    This closed-form solution is valid for a symmetric binary (1:1)
    electrolyte only. It is not valid for mixed-valence or multi-component
    electrolytes.
    """

    def __init__(
        self,
        edl_obj: ElectricalDoubleLayer,
        x: unxt.Quantity,
        sigma: unxt.Quantity,
    ):
        super().__init__(edl_obj=edl_obj, x=x, sigma=sigma)
        self.bulk_concentration, self.valancy = (
            self._validate_symmetric_binary_electrolyte()
        )

    def _validate_symmetric_binary_electrolyte(self) -> unxt.Quantity:
        ions = self.edl_obj.electrolyte.ions
        if len(ions) != 2:
            raise ValueError(
                "NonLinearPoissonBoltzmann analytical solution requires exactly "
                f"2 ions (symmetric z:z electrolyte), got {len(ions)}."
            )

        z0 = ions[0].charge.to("C").value / _constants.ELEMENTARY_CHARGE.to("C").value
        z1 = ions[1].charge.to("C").value / _constants.ELEMENTARY_CHARGE.to("C").value

        # symmetric binary electrolyte: valences must be +1 and -1
        if not (jnp.isclose(z0 + z1, 0.0)):
            raise ValueError(
                "NonLinearPoissonBoltzmann analytical solution requires a "
                "symmetric z:z electrolyte."
            )

        c0 = ions[0].molar_conc.to("mol/L")
        c1 = ions[1].molar_conc.to("mol/L")
        if not jnp.isclose((c0 - c1).to("mol/L").value, 0.0):
            raise ValueError(
                "NonLinearPoissonBoltzmann analytical solution requires equal "
                "bulk concentrations for cation and anion."
            )

        return c0, jnp.abs(z0)  # return bulk concentration and valence magnitude

    @property
    def alpha(self):
        """Inverse of the Grahame relation prefactor."""
        return 1.0 / jnp.sqrt(
            8
            * _constants.BOLTZMANN_CONSTANT
            * self.edl_obj.temperature
            * self.edl_obj.electrolyte.epsilon
            * self.bulk_concentration
            * _constants.AVOGADRO_NUMBER
        )

    @property
    def a(self):
        """Intermediate quantity for the non-linear PB closed-form solution."""
        sigma_alpha = (self.sigma * self.alpha).to(" ")
        return (jnp.sqrt(sigma_alpha**2 + 1) - 1) / sigma_alpha * self.exp_factor

    @property
    def phi(self):
        """Non-linear PB potential profile in Volts."""
        _phi = (
            unxt.Quantity(jnp.arctanh(self.a).value, unit=" ")
            * 4
            * (_constants.BOLTZMANN_CONSTANT * self.edl_obj.temperature).to("eV")
        ) / self.valancy
        return unxt.Quantity(_phi.value, "V")


if __name__ == "__main__":
    edl_obj = ElectricalDoubleLayer("input.json")
    x = unxt.Quantity(jnp.linspace(0, 50.0, 500), unit="nm")
    sigma = _constants.ELEMENTARY_CHARGE / unxt.Quantity(1e4, "angstrom^2")

    linear_pb = LinearPoissonBoltzmann(edl_obj=edl_obj, x=x, sigma=sigma)
    # potential in V
    print(linear_pb.phi)
    # charge density in mol/L
    print(linear_pb.rho)
    c_cation = edl_obj.electrolyte.ions[0].molar_conc * boltzmann_factor(
        phi=linear_pb.phi, temperature=edl_obj.temperature, valency=1
    )
    c_anion = edl_obj.electrolyte.ions[1].molar_conc * boltzmann_factor(
        phi=linear_pb.phi, temperature=edl_obj.temperature, valency=-1
    )
    # ion conc in mol/L
    print(c_cation, c_anion)

    non_linear_pb = NonLinearPoissonBoltzmann(edl_obj=edl_obj, x=x, sigma=sigma)
    # potential in V
    print(non_linear_pb.phi)
    c_cation = edl_obj.electrolyte.ions[0].molar_conc * boltzmann_factor(
        phi=non_linear_pb.phi, temperature=edl_obj.temperature, valency=1
    )
    c_anion = edl_obj.electrolyte.ions[1].molar_conc * boltzmann_factor(
        phi=non_linear_pb.phi, temperature=edl_obj.temperature, valency=-1
    )
    # ion conc in mol/L
    print(c_cation, c_anion)
