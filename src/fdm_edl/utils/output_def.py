# SPDX-License-Identifier: GPL-3.0-or-later
from dataclasses import dataclass
from typing import Dict

import jax
import unxt

from .unit_conversion import check_data_type


# for mandatory quantities, set attr
# for optional quantities, set DT | None (default=None)
@dataclass(frozen=True)
class EDLStatus:
    coordinate: unxt.Quantity
    sigma: unxt.Quantity
    phi: unxt.Quantity
    efield: unxt.Quantity | None
    rho: unxt.Quantity | None
    ion_conc: Dict[str, unxt.Quantity]

    def __post_init__(self):
        check_data_type(self.coordinate, "length")
        check_data_type(self.sigma, "surface charge density")
        check_data_type(self.phi, "electrical potential")
        if self.efield is not None:
            check_data_type(self.efield, "electrical field strength")
        if self.rho is not None:
            check_data_type(self.rho, "electrical charge density")
        for conc in self.ion_conc.values():
            check_data_type(conc, "molar concentration")


@dataclass(frozen=True)
class IsothermStatus:
    phi: unxt.Quantity
    coverage: jax.Array
    temperature: unxt.Quantity
    n_et: float
    coverage_max: float = 1.0
    lateral_interaction: unxt.Quantity | None = None

    def __post_init__(self):
        check_data_type(self.phi, "electrical potential")
        check_data_type(self.temperature, "temperature")
        if self.lateral_interaction is not None:
            # energy per mol
            check_data_type(self.lateral_interaction, "chemical potential")
