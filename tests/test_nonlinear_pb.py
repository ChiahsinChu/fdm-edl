# SPDX-License-Identifier: GPL-3.0-or-later
import unittest
from pathlib import Path

import numpy as np
import unxt
from jax import numpy as jnp

from fdm_edl.api import ConstP, ElectricalDoubleLayer, Symmetric
from fdm_edl.benchmark import NonLinearPoissonBoltzmann


class TestNonLinearPoissonBoltzmannSolution(unittest.TestCase):
    def setUp(self) -> None:
        example_input = Path(__file__).resolve().parent / "data" / "CaSO4.json"
        self.edl_obj = ElectricalDoubleLayer(example_input)
        debye_length = self.edl_obj.electrolyte.debye_length

        n_grid = 500
        self.x = unxt.Quantity(
            jnp.linspace(0.0, debye_length.to("nm").value * 10.0, n_grid), unit="nm"
        )
        self.phi_0 = unxt.Quantity(0.025, "V")

        # analytical solution
        self.ref_edl_obj = NonLinearPoissonBoltzmann(edl_obj=self.edl_obj)
        self.ref_edl_obj.compute(x=self.x, phi_0=self.phi_0)

    def test_numerical(self) -> None:
        # Dirichlet BCs: phi(0) = phi_0, phi(L) = 0
        bcs = ()
        bcs += ConstP(phi=self.phi_0)([0])
        bcs += ConstP(phi=unxt.Quantity(0.0, unit=self.phi_0.unit))([self.x.size - 1])

        self.edl_obj.compute(self.x, bcs)

        np.testing.assert_allclose(
            self.ref_edl_obj.edl_status.phi.value,
            self.edl_obj.result.phi.value,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_consisistency(self) -> None:
        # Check that the reference solution is self-consistent.
        # Dirichlet BCs: phi(0) = phi_0, phi(L) = 0
        bcs = ()
        bcs += ConstP(phi=self.phi_0)([0])
        bcs += ConstP(phi=unxt.Quantity(0.0, unit=self.phi_0.unit))([self.x.size - 1])

        self.edl_obj.compute(self.x, bcs)
        phi_1 = self.edl_obj.result.phi.value

        # Dirichlet BC: phi(0) = phi_0; Neumann BC: dphi/dx(L) = 0
        bcs = ()
        bcs += ConstP(phi=self.phi_0)([0])
        bcs += Symmetric()([self.x.size - 1])

        self.edl_obj.compute(self.x, bcs)
        phi_2 = self.edl_obj.result.phi.value

        np.testing.assert_allclose(
            phi_1,
            phi_2,
            atol=1e-5,
            rtol=1e-5,
        )
