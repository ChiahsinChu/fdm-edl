# SPDX-License-Identifier: GPL-3.0-or-later
import unittest
from pathlib import Path

import jax
import unxt
from jax import numpy as jnp

from fdm_edl.api import ElectricalDoubleLayer
from fdm_edl.benchmark.pb1d import NonLinearPoissonBoltzmann
from fdm_edl.solver.grad.laplacian import LaplacianOP
from fdm_edl.utils import constants
from fdm_edl.utils import unit_conversion as uc


class GradientTester:
    def test_numerical(self) -> None:
        phi = self._func(self.x)

        ad_grad = jax.vmap(jax.grad(self._func))(self.x)
        ad_lap = jax.vmap(jax.hessian(self._func))(self.x)
        fd_grad, fd_div_D = self.grad_op(self.x.value, phi)

        # Check that the AD and FD gradients are close.
        self.assertTrue(
            jnp.allclose(
                ad_grad.value,
                fd_grad,
                atol=1e-5,
                rtol=1e-5,
            )
        )

        eps_0 = constants.VACUUM_PERMITTIVITY.to(
            uc.UNIT_SYSTEMS["metal"]["permittivity"]
        ).value
        eps = self.grad_op.eps_func(jnp.abs(fd_grad)) * eps_0
        self.assertTrue(
            jnp.allclose(
                ad_lap.value.value,
                -fd_div_D / eps,
                atol=1e-5,
                rtol=1e-5,
            )
        )


class TestLaplacianOP(unittest.TestCase, GradientTester):
    def setUp(self) -> None:
        example_input = Path(__file__).resolve().parent / "data" / "CaSO4.json"
        edl_obj = ElectricalDoubleLayer(example_input)
        non_linear_pb = NonLinearPoissonBoltzmann(edl_obj=edl_obj)

        phi_0 = unxt.Quantity(0.025, "V")
        debye_length = non_linear_pb.edl_obj.electrolyte.debye_length.to("angstrom")
        beta = non_linear_pb.beta

        def _func(x):
            return non_linear_pb.compute_phi(
                x,
                phi_0,
                debye_length,
                beta,
                non_linear_pb.valency,
            ).value

        self.x = unxt.Quantity(jnp.linspace(0, 50.0, 500), unit="angstrom")
        self._func = _func
        self.grad_op = LaplacianOP()
