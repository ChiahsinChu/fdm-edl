# SPDX-License-Identifier: GPL-3.0-or-later
import unittest
from pathlib import Path

import numpy as np
import quaxed.numpy as jnp
import unxt

from fdm_edl import _constants
from fdm_edl.bc import DirichletBC
from fdm_edl.benchmark import NonLinearPoissonBoltzmann
from fdm_edl.edl import ElectricalDoubleLayer


class TestNonLinearPoissonBoltzmannSolution(unittest.TestCase):
    def setUp(self) -> None:
        example_input = Path(__file__).resolve().parent / "data" / "NaCl.json"
        self.edl_obj = ElectricalDoubleLayer(example_input)

    def test_numerical(self) -> None:
        n_grid = 500
        x = unxt.Quantity(jnp.linspace(0.0, 50.0, n_grid), unit="nm")
        sigma = _constants.ELEMENTARY_CHARGE / unxt.Quantity(1e4, "angstrom^2")

        # analytical solution
        non_linear_pb = NonLinearPoissonBoltzmann(edl_obj=self.edl_obj)
        non_linear_pb.compute(x=x, sigma=sigma)
        # numerical solution
        phi_wall = non_linear_pb.edl_status.phi[0]
        boundary_conditions = [
            DirichletBC(
                [0],
                unxt.Quantity(jnp.full((1,), phi_wall.value), unit=phi_wall.unit),
            ),
            DirichletBC(
                [n_grid - 1],
                unxt.Quantity(jnp.zeros((1,)), unit=phi_wall.unit),
            ),
        ]
        self.edl_obj.compute(x, boundary_conditions)

        np.testing.assert_allclose(
            non_linear_pb.edl_status.phi.value,
            self.edl_obj.result.solution.value,
            atol=1e-5,
            rtol=1e-5,
        )
