# SPDX-License-Identifier: GPL-3.0-or-later
import unittest
from pathlib import Path

import jax
import unxt
from jax import numpy as jnp

from fdm_edl.api import ElectricalDoubleLayer
from fdm_edl.benchmark.pb1d import NonLinearPoissonBoltzmann
from fdm_edl.op import EuclideanFDOp, EuclideanFVOp


class GradientOpTester:
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

        eps = self.grad_op.eps_func(jnp.abs(fd_grad))
        self.assertTrue(
            jnp.allclose(
                ad_lap.value.value,
                -fd_div_D / eps,
                atol=1e-5,
                rtol=1e-5,
            )
        )


class TestEuclideanFDOp(unittest.TestCase, GradientOpTester):
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
        self.grad_op = EuclideanFDOp()


class TestEuclideanFVOp(unittest.TestCase, GradientOpTester):
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

        def eps_func(coord):
            return 1.0 + jax.nn.sigmoid(coord) * 78.4

        # use position-dependent permittivity to test eps_func handling in FV operator
        self.grad_op = EuclideanFVOp(eps_func=eps_func)


# class TestCylindricalGradientOps(unittest.TestCase):
#     def test_finite_difference_cylindrical_operator(self) -> None:
#         x = jnp.linspace(0.05, 3.0, 800)
#         y = x**2

#         op_cart = EuclideanFDOp(
#             uniform=True,
#             boundary_points=5,
#             interior_points=5,
#             eps_func=lambda e: jnp.ones_like(e),
#             coordinate_system="cartesian",
#         )
#         op_cyl = EuclideanFDOp(
#             uniform=True,
#             boundary_points=5,
#             interior_points=5,
#             eps_func=lambda e: jnp.ones_like(e),
#             coordinate_system="cylindrical",
#         )

#         _, div_cart = op_cart(x, y)
#         _, div_cyl = op_cyl(x, y)

#         eps_0 = constants.VACUUM_PERMITTIVITY.to(
#             uc.UNIT_SYSTEMS["metal"]["permittivity"]
#         ).value
#         lap_cart = -div_cart / eps_0
#         lap_cyl = -div_cyl / eps_0

#         # Ignore boundary rows where one-sided stencils are used.
#         self.assertTrue(jnp.allclose(lap_cart[5:-5], 2.0, atol=3e-3, rtol=3e-3))
#         self.assertTrue(jnp.allclose(lap_cyl[5:-5], 4.0, atol=8e-3, rtol=8e-3))

#     def test_finite_volume_cylindrical_operator(self) -> None:
#         x = jnp.linspace(0.05, 3.0, 800)
#         y = x**2

#         op_cart = EuclideanFVOp(
#             eps_func=lambda e: jnp.ones_like(e),
#             coordinate_system="cartesian",
#         )
#         op_cyl = EuclideanFVOp(
#             eps_func=lambda e: jnp.ones_like(e),
#             coordinate_system="cylindrical",
#         )

#         _, div_cart = op_cart(x, y)
#         _, div_cyl = op_cyl(x, y)

#         self.assertTrue(jnp.allclose(div_cart[5:-5], -2.0, atol=5e-6, rtol=5e-6))
#         self.assertTrue(jnp.allclose(div_cyl[5:-5], -4.0, atol=5e-6, rtol=5e-6))
