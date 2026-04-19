# SPDX-License-Identifier: GPL-3.0-or-later
import unittest
from pathlib import Path

import numpy as np
import quaxed.numpy as jnp
import unxt

from fdm_edl.api import ConstP, ElectricalDoubleLayer, Stern
from fdm_edl.benchmark import GCSModel


class TestZZGCS(unittest.TestCase):
    def setUp(self) -> None:
        example_input = Path(__file__).resolve().parent / "data" / "CaSO4.json"
        self.edl_obj = ElectricalDoubleLayer(example_input)

        self.d_ohp = unxt.Quantity(5.0, "Angstrom")
        self.eps_ohp = 6.0
        self.phi_0 = unxt.Quantity(0.025, "V")

        debye_length = self.edl_obj.electrolyte.debye_length
        n_grid = 600
        self.x = (
            unxt.Quantity(
                jnp.linspace(0.0, 10.0 * debye_length.to("nm").value, n_grid), unit="nm"
            )
            + self.d_ohp
        )

        self.ref_edl_obj = GCSModel(
            edl_obj=self.edl_obj,
            d_ohp=self.d_ohp,
            eps_ohp=self.eps_ohp,
        )
        self.ref_edl_obj.compute(x=self.x, phi_0=self.phi_0)

    def test_numerical(self) -> None:
        bcs = ()
        bcs += Stern(
            phi=self.phi_0,
            eps_gc=self.edl_obj.electrolyte.epsilon_r,
            eps_s=self.eps_ohp,
            d_s=self.d_ohp,
        )([0])
        bcs += ConstP(phi=unxt.Quantity(0.0, unit=self.phi_0.unit))([self.x.size - 1])

        self.edl_obj.compute(self.x, bcs)

        np.testing.assert_allclose(
            self.edl_obj.result.phi.value,
            self.ref_edl_obj.edl_status.phi.value,
            atol=5e-5,
            rtol=5e-5,
        )
