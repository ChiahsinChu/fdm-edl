# SPDX-License-Identifier: GPL-3.0-or-later
from pathlib import Path
import unittest

import numpy as np
import quaxed.numpy as jnp
import unxt

from fdm_edl import _constants
from fdm_edl.edl import ElectricalDoubleLayer
from fdm_edl.test import NonLinearPoissonBoltzmann


class TestNonLinearPoissonBoltzmannSolution(unittest.TestCase):
    def setUp(self) -> None:
        example_input = Path(__file__).resolve().parent / "data" / "NaCl.json"
        self.edl_obj = ElectricalDoubleLayer(example_input)
    
    def test_numerical(self) -> None:
        x = unxt.Quantity(jnp.linspace(0.0, 50.0, 500), unit="nm")
        sigma = _constants.ELEMENTARY_CHARGE / unxt.Quantity(1e4, "angstrom^2")

        # analytical solution
        non_linear_pb = NonLinearPoissonBoltzmann(edl_obj=self.edl_obj, x=x, sigma=sigma)
        # numerical solution
        self.edl_obj.compute(x, non_linear_pb.phi[0])

        np.testing.assert_allclose(
            non_linear_pb.phi.value,
            self.edl_obj.result.solution.value,
            atol=1e-5,
            rtol=1e-5,
        )
