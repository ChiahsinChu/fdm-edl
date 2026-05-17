# SPDX-License-Identifier: GPL-3.0-or-later
import unittest
from pathlib import Path

import numpy as np
import unxt
from jax import numpy as jnp

from fdm_edl.api import ConstP, ElectricalDoubleLayer
from fdm_edl.benchmark import NonLinearPoissonBoltzmann
from fdm_edl.utils import load_dict


class SolverTester:
    def test_numerical(self) -> None:
        _params = load_dict(Path(__file__).resolve().parent / "data" / "CaSO4.json")
        _params["solver"] = self.solver_params

        edl_obj = ElectricalDoubleLayer(_params)
        debye_length = edl_obj.electrolyte.debye_length

        n_grid = 500
        x = unxt.Quantity(
            jnp.linspace(0.0, debye_length.to("nm").value * 10.0, n_grid), unit="nm"
        )
        phi_0 = unxt.Quantity(0.025, "V")

        # analytical solution
        ref_edl_obj = NonLinearPoissonBoltzmann(edl_obj=edl_obj)
        ref_edl_obj.compute(x=x, phi_0=phi_0)

        # Dirichlet BCs: phi(0) = phi_0, phi(L) = 0
        bcs = ()
        bcs += ConstP(phi=phi_0)([0])
        bcs += ConstP(phi=unxt.Quantity(0.0, unit=phi_0.unit))([x.size - 1])

        edl_obj.compute(x, bcs)

        np.testing.assert_allclose(
            ref_edl_obj.edl_status.phi.value,
            edl_obj.result.phi.value,
            atol=1e-5,
            rtol=1e-5,
        )


class TestNewtonSolver(unittest.TestCase, SolverTester):
    def setUp(self) -> None:
        self.solver_params = {
            "method": "newton",
            "max_iter": 500,
            "atol_var": 1e-6,
        }


class TestBiCGStabSolver(unittest.TestCase, SolverTester):
    def setUp(self) -> None:
        self.solver_params = {
            "method": "bicgstab",
            "max_iter": 500,
            "atol_var": 1e-7,
            "atol_res": 1e-6,
        }


class TestCGSolver(unittest.TestCase, SolverTester):
    def setUp(self) -> None:
        self.solver_params = {
            "method": "cg",
            "max_iter": 500,
            "atol_var": 1e-7,
            "atol_res": 1e-6,
        }


class TestGMRESSolver(unittest.TestCase, SolverTester):
    def setUp(self) -> None:
        self.solver_params = {
            "method": "gmres",
            "max_iter": 500,
            "atol_var": 1e-7,
            "atol_res": 1e-6,
        }
