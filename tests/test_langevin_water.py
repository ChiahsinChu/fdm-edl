# SPDX-License-Identifier: GPL-3.0-or-later
import unittest

import jax
import quaxed.numpy as qjnp
import unxt
from jax import numpy as jnp

from fdm_edl.utils.langevin import LangevinWaterEps


class TestLangevinWaterEps(unittest.TestCase):
    def setUp(self) -> None:
        self.temperature = unxt.Quantity(300, "K")

        self.eps_obj = LangevinWaterEps(self.temperature)
        self._efields = jnp.logspace(-10, 10, 10)
        self.efields = unxt.Quantity(self._efields, "V/angstrom")
        self.eps = self.eps_obj(self.efields)

    def test_zero_limit(self) -> None:
        # Check eps converge to the zero-field value in the limit of zero field.
        self.assertTrue(jnp.isclose(self.eps[0], self.eps_obj.eps_static))
        # Check that the gradient of eps with respect to field is zero at zero field.
        self.assertTrue(
            jnp.isclose(jax.grad(self.eps_obj.compute)(jnp.array(0.0)), 0.0)
        )

    def test_high_field_limit(self) -> None:
        # Check that eps converges to the optical value in the limit of high field.
        self.assertTrue(jnp.isclose(self.eps[-1], self.eps_obj.eps_opt))
        # Check that the gradient of eps with respect to field is zero at high field.
        self.assertTrue(
            jnp.isclose(jax.grad(self.eps_obj.compute)(self._efields[-1]), 0.0)
        )

    def test_monotonicity(self) -> None:
        # Check that eps is a monotonically decreasing function of field.
        self.assertTrue(jnp.all(jnp.diff(self.eps) <= 0.0))

    def test_batch(self) -> None:
        # Check that the batch and non-batch versions of the gradient give the same result.
        grad_batch = jax.vmap(jax.grad(self.eps_obj.compute))(self._efields)
        for g, e in zip(grad_batch, self._efields):
            grad_single = jax.grad(self.eps_obj.compute)(e)
            self.assertTrue(jnp.isclose(g, grad_single))

    def test_unxt(self) -> None:
        self.assertTrue(
            jnp.isclose(
                jax.grad(self.eps_obj)(
                    unxt.Quantity(qjnp.array(0.0), "V/angstrom")
                ).value,
                0.0,
            )
        )
        self.assertTrue(
            jnp.isclose(
                jax.grad(self.eps_obj)(self.efields[0]).value,
                0.0,
            )
        )
        self.assertTrue(
            jnp.allclose(
                jax.vmap(jax.grad(self.eps_obj.compute))(self._efields),
                jax.vmap(jax.grad(self.eps_obj))(self.efields).value,
            )
        )
