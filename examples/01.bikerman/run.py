# SPDX-License-Identifier: GPL-3.0-or-later
import unxt
from jax import numpy as jnp

from fdm_edl.api import ElectricalDoubleLayer
from fdm_edl.api.bc import ConstP, Stern

x_img = unxt.Quantity(0.9, "angstrom")
x_IHP = unxt.Quantity(2.7, "angstrom")
x_OHP = unxt.Quantity(4.5, "angstrom")


eps_s = 6.0
eps_gc = 78.4

n_grid = 500
edl_obj = ElectricalDoubleLayer("input.json")
debye_length = edl_obj.electrolyte.debye_length
x = (
    unxt.Quantity(
        jnp.linspace(0.0, debye_length.to("angstrom").value * 10.0, n_grid),
        unit="angstrom",
    )
    + x_OHP
)

phi0_guess = None
for phi in unxt.Quantity(jnp.linspace(-1.0, 1.0, 10), unit="V"):
    bcs = ()
    bcs += Stern(
        phi=phi,
        eps_gc=eps_gc,
        eps_s=eps_s,
        d_s=(x_OHP - x_IHP),
    )([0])
    bcs += ConstP(unxt.Quantity(0.0, "V"))([-1])

    edl_obj.compute(x, bcs, phi0=phi0_guess)
    assert edl_obj.result is not None
    phi0_guess = edl_obj.result.phi

    # edl_obj.compute_residual(jnp.ones_like(x.value), bcs, x.value)
    print(phi, edl_obj.result.sigma)
