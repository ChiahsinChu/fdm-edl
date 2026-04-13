# SPDX-License-Identifier: GPL-3.0-or-later
import unittest
from pathlib import Path

import numpy as np
import quaxed.numpy as jnp
import unxt

from fdm_edl.api import ElectricalDoubleLayer
from fdm_edl.benchmark import NonLinearPoissonBoltzmann
from fdm_edl.utils.bc import BoundaryCondition


class TestNonLinearPoissonBoltzmannSolution(unittest.TestCase):
    def setUp(self) -> None:
        example_input = Path(__file__).resolve().parent / "data" / "CaSO4.json"
        self.edl_obj = ElectricalDoubleLayer(example_input)

        n_grid = 500
        self.x = unxt.Quantity(jnp.linspace(0.0, 10.0, n_grid), unit="nm")
        self.phi_0 = unxt.Quantity(0.025, "V")

        # analytical solution
        self.ref_edl_obj = NonLinearPoissonBoltzmann(edl_obj=self.edl_obj)
        self.ref_edl_obj.compute(x=self.x, phi_0=self.phi_0)

    def test_consisistency(self) -> None:
        # Check that the reference solution is self-consistent.
        # Dirichlet BCs: phi(0) = phi_0, phi(L) = 0
        boundary_conditions = [
            BoundaryCondition(
                1.0,
                0.0,
                -self.phi_0,
                [0],
            ),
            BoundaryCondition(
                1.0,
                0.0,
                unxt.Quantity(0.0, unit=self.phi_0.unit),
                [self.x.size - 1],
            ),
        ]

        self.edl_obj.compute(self.x, boundary_conditions)
        phi_1 = self.edl_obj.result.phi.value

        # Dirichlet BC: phi(0) = phi_0; Neumann BC: dphi/dx(L) = 0
        boundary_conditions = [
            BoundaryCondition(
                1.0,
                0.0,
                -self.phi_0,
                [0],
            ),
            BoundaryCondition(
                0.0,
                1.0,
                unxt.Quantity(0.0, unit=self.phi_0.unit),
                [self.x.size - 1],
            ),
        ]

        self.edl_obj.compute(self.x, boundary_conditions)
        phi_2 = self.edl_obj.result.phi.value

        np.testing.assert_allclose(
            phi_1,
            phi_2,
            atol=1e-5,
            rtol=1e-5,
        )

    # def test_numerical_dirichlet(self) -> None:
    #     # Dirichlet BCs: phi(0) = phi_0, phi(L) = 0
    #     boundary_conditions = [
    #         BoundaryCondition(
    #             1.0,
    #             0.0,
    #             -self.phi_0,
    #             [0],
    #         ),
    #         BoundaryCondition(
    #             1.0,
    #             0.0,
    #             unxt.Quantity(0.0, unit=self.phi_0.unit),
    #             [self.x.size - 1],
    #         ),
    #     ]

    #     self.edl_obj.compute(self.x, boundary_conditions)

    #     np.testing.assert_allclose(
    #         self.ref_edl_obj.edl_status.phi.value,
    #         self.edl_obj.result.phi.value,
    #         atol=1e-5,
    #         rtol=1e-5,
    #     )

    # def test_numerical_neumann(self) -> None:
    #     # Dirichlet BC: phi(0) = phi_0; Neumann BC: dphi/dx(L) = 0
    #     boundary_conditions = [
    #         BoundaryCondition(
    #             1.0,
    #             0.0,
    #             -self.phi_0,
    #             [0],
    #         ),
    #         BoundaryCondition(
    #             0.0,
    #             1.0,
    #             unxt.Quantity(0.0, unit=self.phi_0.unit),
    #             [self.x.size - 1],
    #         ),
    #     ]

    #     self.edl_obj.compute(self.x, boundary_conditions)

    #     np.testing.assert_allclose(
    #         self.ref_edl_obj.edl_status.phi.value,
    #         self.edl_obj.result.phi.value,
    #         atol=1e-5,
    #         rtol=1e-5,
    #     )
