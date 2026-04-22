# SPDX-License-Identifier: GPL-3.0-or-later
from typing import cast

import jax
import quaxed as qjax
import quaxed.numpy as jnp
import unxt
from astropy.units import cds  # type: ignore[import-untyped]

from ..api import ElectricalDoubleLayer
from ..models.base import boltzmann_factor
from ..utils import constants
from ..utils.output_def import EDLStatus


@jax.jit
def _linear_exponent(x: unxt.Quantity, debye_length: unxt.Quantity) -> unxt.Quantity:
    """Dimensionless exponential decay factor exp(-x / λ_D)."""
    return cast(unxt.Quantity, jnp.exp(-(x / debye_length)))


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
        self.beta = 1.0 / (constants.BOLTZMANN_CONSTANT * self.edl_obj.temperature).to(
            "eV"
        )

        # placeholders for results to be computed in compute()
        self.edl_status: EDLStatus | None = None

    def phi0_to_sigma(
        self,
        phi_0: unxt.Quantity,
    ) -> unxt.Quantity:
        """Convert surface potential to surface charge density.

        Parameters
        ----------
        phi_0 : unxt.Quantity
            Surface potential at x = 0.

        Returns
        -------
        unxt.Quantity
            Surface charge density.
        """
        raise NotImplementedError("Subclasses must implement phi0_to_sigma().")

    def compute(
        self,
        x: unxt.Quantity,
        phi_0: unxt.Quantity,
    ):
        """Evaluate the analytical PB solution.

        Parameters
        ----------
        x : unxt.Quantity
            Spatial coordinates at which to evaluate the solution.
        phi_0 : unxt.Quantity
            Surface potential at x = 0.
        """
        raise NotImplementedError("Subclasses must implement the compute() method.")


class LinearPoissonBoltzmann(BasePoissonBoltzmann):
    """Analytical linearized Poisson-Boltzmann profile in 1D.

    Users provide an ElectricalDoubleLayer object, a surface potential
    phi_0, and spatial coordinates.
    """

    def phi0_to_sigma(
        self,
        phi_0: unxt.Quantity,
    ) -> unxt.Quantity:
        """sigma = phi_0 * epsilon / lambda_D."""
        epsilon = self.edl_obj.electrolyte.solvent.eps_0 * constants.VACUUM_PERMITTIVITY
        debye_length = self.edl_obj.electrolyte.debye_length
        return cast(
            unxt.Quantity,
            (phi_0 * epsilon / debye_length).to(cds.e / unxt.unit("angstrom^2")),
        )

    @staticmethod
    @jax.jit
    def compute_phi(x, phi_0, debye_length):
        return phi_0 * _linear_exponent(x, debye_length)

    def compute(
        self,
        x: unxt.Quantity,
        phi_0: unxt.Quantity,
    ):
        """Evaluate the linearised PB solution.

        Parameters
        ----------
        x : unxt.Quantity
            Spatial coordinates at which to evaluate the solution.
        phi_0 : unxt.Quantity
            Surface potential at x = 0.
        """
        _x = x.to("angstrom")
        debye_length = self.edl_obj.electrolyte.debye_length.to("angstrom")
        sigma = self.phi0_to_sigma(phi_0)

        phi = self.compute_phi(_x, phi_0, debye_length).to("V")
        compute_e_field = qjax.vmap(
            qjax.grad(self.compute_phi, argnums=0), in_axes=(0, None, None)
        )
        efield = -compute_e_field(_x, phi_0, debye_length).to("V/m")

        # Linearised PB charge density profile [mol/L].
        epsilon = self.edl_obj.electrolyte.solvent.eps_0 * constants.VACUUM_PERMITTIVITY
        _rho = (phi_0 * epsilon / debye_length**2) * _linear_exponent(_x, debye_length)
        rho = (_rho / (constants.ELEMENTARY_CHARGE * constants.AVOGADRO_NUMBER)).to(
            "mol/L"
        )

        self.edl_status = EDLStatus(
            coordinate=x.to("angstrom"),
            sigma=sigma,
            phi=phi,
            efield=efield,
            rho=rho,
            ion_conc={
                name: ion.molar_conc
                * boltzmann_factor(
                    phi=phi,
                    temperature=self.edl_obj.temperature,
                    charge=ion.charge,
                )
                for name, ion in self.edl_obj.electrolyte.ions.items()
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

    def _validate_symmetric_binary_electrolyte(
        self,
    ) -> tuple[unxt.Quantity, unxt.Quantity]:
        ions = self.edl_obj.electrolyte.ions
        if len(ions) != 2:
            raise ValueError(
                "NonLinearPoissonBoltzmann analytical solution requires exactly "
                f"2 ions (symmetric z:z electrolyte), got {len(ions)}."
            )

        z_tot = 0.0
        zi = 0.0
        for _name, ion in ions.items():
            zi = float(
                ion.charge.to("C").value / constants.ELEMENTARY_CHARGE.to("C").value
            )
            z_tot += zi

        # symmetric binary electrolyte: valences must be +1 and -1
        if not (jnp.isclose(z_tot, 0.0)):
            raise ValueError(
                "NonLinearPoissonBoltzmann analytical solution requires a "
                "symmetric z:z electrolyte."
            )

        c0 = ion.molar_conc.to("mol/L")

        return c0, unxt.Quantity(
            jnp.abs(zi), unit=cds.e
        )  # return bulk concentration and valence magnitude

    def phi0_to_sigma(
        self,
        phi_0: unxt.Quantity,
    ) -> unxt.Quantity:
        """Grahame equation: sigma = sqrt(8 eps k_B T c_0 N_A) sinh(z e phi_0 / 2 k_B T)."""
        epsilon = self.edl_obj.electrolyte.solvent.eps_0 * constants.VACUUM_PERMITTIVITY
        grahame_prefactor = jnp.sqrt(
            8
            * constants.BOLTZMANN_CONSTANT
            * self.edl_obj.temperature
            * epsilon
            * self.bulk_concentration
            * constants.AVOGADRO_NUMBER
        )
        arg = (self.valency * self.beta * phi_0 / 2).to("")
        return (grahame_prefactor * jnp.sinh(arg)).to(cds.e / unxt.unit("angstrom^2"))

    @staticmethod
    @jax.jit
    def compute_phi(x, phi_0, debye_length, beta, valency):
        gamma = jnp.tanh(beta * valency * phi_0 / 4)
        return 4 * jnp.arctanh(gamma * jnp.exp(-(x / debye_length))) / beta / valency

    def compute(
        self,
        x: unxt.Quantity,
        phi_0: unxt.Quantity,
    ):
        """Evaluate the non-linear PB solution.

        Parameters
        ----------
        x : unxt.Quantity
            Spatial coordinates at which to evaluate the solution.
        phi_0 : unxt.Quantity
            Surface potential at x = 0.
        """
        _x = x.to("angstrom")
        debye_length = self.edl_obj.electrolyte.debye_length.to("angstrom")
        beta = self.beta
        sigma = self.phi0_to_sigma(phi_0)

        _phi = self.compute_phi(
            _x,
            phi_0,
            debye_length,
            beta,
            self.valency,
        )
        phi = unxt.Quantity(_phi.value, unit="V")
        efield = -qjax.vmap(
            qjax.grad(self.compute_phi), in_axes=(0, None, None, None, None)
        )(
            _x,
            phi_0,
            debye_length,
            beta,
            self.valency,
        )
        self.edl_status = EDLStatus(
            coordinate=x.to("angstrom"),
            sigma=sigma,
            phi=phi,
            efield=efield.to("V/Angstrom"),
            rho=None,
            ion_conc={
                name: ion.molar_conc
                * boltzmann_factor(
                    phi=phi,
                    temperature=self.edl_obj.temperature,
                    charge=ion.charge,
                )
                for name, ion in self.edl_obj.electrolyte.ions.items()
            },
        )


if __name__ == "__main__":
    edl_obj = ElectricalDoubleLayer("input.json")
    x = unxt.Quantity(jnp.linspace(0, 50.0, 500), unit="nm")
    phi_0 = unxt.Quantity(0.025, "V")

    linear_pb = LinearPoissonBoltzmann(edl_obj=edl_obj)
    linear_pb.compute(x=x, phi_0=phi_0)
    assert linear_pb.edl_status is not None
    # potential in V
    print(linear_pb.edl_status.phi)
    # sigma derived from phi_0
    print(linear_pb.edl_status.sigma)
    # charge density in mol/L
    print(linear_pb.edl_status.rho)
    # ion conc in mol/L
    ion_names = list(edl_obj.electrolyte.ions)
    c_cation = linear_pb.edl_status.ion_conc[ion_names[0]]
    c_anion = linear_pb.edl_status.ion_conc[ion_names[1]]
    print(c_cation, c_anion)

    non_linear_pb = NonLinearPoissonBoltzmann(edl_obj=edl_obj)
    non_linear_pb.compute(x=x, phi_0=phi_0)
    assert non_linear_pb.edl_status is not None
    # potential in V
    print(non_linear_pb.edl_status.phi)
    # sigma derived from Grahame equation
    print(non_linear_pb.edl_status.sigma)
    # ion conc in mol/L
    c_cation = non_linear_pb.edl_status.ion_conc[ion_names[0]]
    c_anion = non_linear_pb.edl_status.ion_conc[ion_names[1]]
    print(c_cation, c_anion)
