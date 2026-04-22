# SPDX-License-Identifier: GPL-3.0-or-later
import unittest
from pathlib import Path

import jax
import unxt
from jax import numpy as jnp

from fdm_edl.api import ElectricalDoubleLayer
from fdm_edl.api.bc import ConstP
from fdm_edl.utils import constants
from fdm_edl.utils import unit_conversion as uc


edl_obj = ElectricalDoubleLayer("./data/NaCl_langevin.json")
debye_length = edl_obj.electrolyte.debye_length
n_grid = 500
x = unxt.Quantity(
    jnp.linspace(0.0, debye_length.to("nm").value * 10.0, n_grid), unit="nm"
)
phi_0 = unxt.Quantity(0.025, "V")


bcs = ()
ConstP(phi=phi_0)([0])
bcs += ConstP(phi=phi_0)([0])
bcs += ConstP(phi=unxt.Quantity(0.0, unit=phi_0.unit))([x.size - 1])

edl_obj.compute(x, bcs)
print(edl_obj._solver_result.converged)
