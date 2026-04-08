# SPDX-License-Identifier: GPL-3.0-or-later
import jax
import quaxed as qjax
import quaxed.numpy as jnp
import unxt
from astropy.units import cds

from .. import _constants
from ..edl import ElectricalDoubleLayer
from ..utils.output import EDLStatus
from .base import boltzmann_factor


@jax.jit
def _linear_exponent(x: unxt.Quantity, debye_length: unxt.Quantity) -> unxt.Quantity:
    """Dimensionless exponential decay factor exp(-x / λ_D)."""
    return jnp.exp(-(x / debye_length))


# todo: control phi(0) via rather than sigma


class BasePoissonBoltzmann:
    """Shared utilities for analytical Poisson-Boltzmann models in 1D."""

    def __init__(
        self,
        edl_obj: ElectricalDoubleLayer,
    ):
        """Initialise the analytical PB model.

        Parameters
        ----------
        edl_obj : ElectricalDoubleLayer
            Configured EDL system (electrolyte, temperature, etc.).
        """
        self.edl_obj = edl_obj
        self.beta = 1.0 / (_constants.BOLTZMANN_CONSTANT * self.edl_obj.temperature).to(
            "eV"
        )

        # placeholders for results to be computed in compute()
        self.edl_status = None

    def compute(
        self,
        x: unxt.Quantity,
        sigma: unxt.Quantity,
    ):
        """...

        Parameters
        ----------
        x : unxt.Quantity
            Spatial coordinates at which to evaluate the solution.
        sigma : unxt.Quantity
            Surface charge density.
        """
        raise NotImplementedError("Subclasses must implement the compute() method.")


class LinearPoissonBoltzmann(BasePoissonBoltzmann):
    """Analytical linearized Poisson-Boltzmann profile in 1D.

    Users provide an ElectricalDoubleLayer object and can optionally provide
    custom spatial coordinates and surface charge density.
    """

    @staticmethod
    @jax.jit
    def compute_phi(x, sigma, debye_length, epsilon):
        exp_factor = _linear_exponent(x, debye_length)
        # Linearised PB potential profile [Volts].
        phi = sigma * debye_length / epsilon * exp_factor
        return phi

    def compute(
        self,
        x: unxt.Quantity,
        sigma: unxt.Quantity,
    ):
        """...

        Parameters
        ----------
        x : unxt.Quantity
            Spatial coordinates at which to evaluate the solution.
        sigma : unxt.Quantity
            Surface charge density.
        """

        _x = x.to("angstrom")
        debye_length = self.edl_obj.electrolyte.debye_length.to("angstrom")
        epsilon = self.edl_obj.electrolyte.epsilon
        phi = self.compute_phi(
            _x,
            sigma,
            debye_length,
            epsilon,
        ).to("V")
        compute_e_field = qjax.vmap(
            qjax.grad(self.compute_phi, argnums=0), in_axes=(0, None, None, None)
        )
        efield = -compute_e_field(_x, sigma, debye_length, epsilon).to("V/m")

        # Linearised PB charge density profile [mol/L].
        _rho = sigma / debye_length * _linear_exponent(_x, debye_length)
        rho = (_rho / (_constants.ELEMENTARY_CHARGE * _constants.AVOGADRO_NUMBER)).to(
            "mol/L"
        )

        self.edl_status = EDLStatus(
            coordinate=x.to("angstrom"),
            sigma=sigma.to(cds.e / unxt.unit("angstrom^2")),
            phi=phi,
            efield=efield,
            rho=rho,
            ion_conc={
                ion.name: ion.molar_conc
                * boltzmann_factor(
                    phi=phi,
                    temperature=self.edl_obj.temperature,
                    valency=int((ion.charge / _constants.ELEMENTARY_CHARGE).to("")),
                )
                for ion in self.edl_obj.electrolyte.ions
            },
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
    ):
        super().__init__(edl_obj=edl_obj)
        self.bulk_concentration, self.valency = (
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

        return c0, unxt.Quantity(
            jnp.abs(z0), unit=cds.e
        )  # return bulk concentration and valence magnitude

    @staticmethod
    @jax.jit
    def compute_phi(x, sigma_alpha, debye_length, beta, valency):
        exp_factor = jnp.exp(-(x / debye_length))
        a = (jnp.sqrt(sigma_alpha**2 + 1) - 1) / sigma_alpha * exp_factor
        # Non-linear PB potential profile [Volts].
        _phi = 4 * jnp.arctanh(a) / beta / valency
        return _phi

    def compute(
        self,
        x: unxt.Quantity,
        sigma: unxt.Quantity,
    ):
        """...

        Parameters
        ----------
        x : unxt.Quantity
            Spatial coordinates at which to evaluate the solution.
        sigma : unxt.Quantity
            Surface charge density.
        """
        _x = x.to("angstrom")
        debye_length = self.edl_obj.electrolyte.debye_length.to("angstrom")
        epsilon = self.edl_obj.electrolyte.epsilon
        # Inverse of the Grahame relation prefactor.
        alpha = 1.0 / jnp.sqrt(
            8
            * _constants.BOLTZMANN_CONSTANT
            * self.edl_obj.temperature
            * epsilon
            * self.bulk_concentration
            * _constants.AVOGADRO_NUMBER
        )
        # Intermediate quantity for the non-linear PB closed-form solution.
        sigma_alpha = (sigma * alpha).to("")
        beta = 1.0 / (self.edl_obj.temperature * _constants.BOLTZMANN_CONSTANT).to("eV")

        _phi = self.compute_phi(
            _x,
            sigma_alpha,
            debye_length,
            beta,
            self.valency,
        )
        phi = unxt.Quantity(_phi.value, unit="V")
        efield = -qjax.vmap(
            qjax.grad(self.compute_phi), in_axes=(0, None, None, None, None)
        )(
            _x,
            sigma_alpha,
            debye_length,
            beta,
            self.valency,
        )
        self.edl_status = EDLStatus(
            coordinate=x.to("angstrom"),
            sigma=sigma.to(cds.e / unxt.unit("angstrom^2")),
            phi=phi,
            efield=efield.to("V/Angstrom"),
            rho=None,
            ion_conc={
                ion.name: ion.molar_conc
                * boltzmann_factor(
                    phi=phi,
                    temperature=self.edl_obj.temperature,
                    valency=int((ion.charge / _constants.ELEMENTARY_CHARGE).to("")),
                )
                for ion in self.edl_obj.electrolyte.ions
            },
        )


if __name__ == "__main__":
    edl_obj = ElectricalDoubleLayer("input.json")
    x = unxt.Quantity(jnp.linspace(0, 50.0, 500), unit="nm")
    sigma = _constants.ELEMENTARY_CHARGE / unxt.Quantity(1e4, "angstrom^2")

    linear_pb = LinearPoissonBoltzmann(edl_obj=edl_obj)
    linear_pb.compute(x=x, sigma=sigma)
    # potential in V
    print(linear_pb.edl_status.phi)
    # charge density in mol/L
    print(linear_pb.edl_status.rho)
    # ion conc in mol/L
    c_cation = linear_pb.edl_status.ion_conc[edl_obj.electrolyte.ions[0].name]
    c_anion = linear_pb.edl_status.ion_conc[edl_obj.electrolyte.ions[1].name]
    print(c_cation, c_anion)

    non_linear_pb = NonLinearPoissonBoltzmann(edl_obj=edl_obj)
    non_linear_pb.compute(x=x, sigma=sigma)
    # potential in V
    print(non_linear_pb.edl_status.phi)
    # ion conc in mol/L
    c_cation = non_linear_pb.edl_status.ion_conc[edl_obj.electrolyte.ions[0].name]
    c_anion = non_linear_pb.edl_status.ion_conc[edl_obj.electrolyte.ions[1].name]
    print(c_cation, c_anion)
