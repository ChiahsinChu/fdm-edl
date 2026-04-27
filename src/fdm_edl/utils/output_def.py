# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import unxt

if TYPE_CHECKING:
    from typing import Dict

    import jax

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
    epsilon_r: jax.Array | None = None

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

    def serialize(self) -> Dict[str, unxt.Quantity | Dict[str, unxt.Quantity]]:
        return {
            "coordinate": self.coordinate.value.tolist(),
            "sigma": self.sigma.value.tolist(),
            "phi": self.phi.value.tolist(),
            "efield": self.efield.value.tolist() if self.efield is not None else None,
            "rho": self.rho.value.tolist() if self.rho is not None else None,
            "ion_conc": {
                key: value.value.tolist() for key, value in self.ion_conc.items()
            },
            "epsilon_r": (
                self.epsilon_r.tolist() if self.epsilon_r is not None else None
            ),
        }

    def deserialize(
        data: Dict[str, unxt.Quantity | Dict[str, unxt.Quantity]],
    ) -> EDLStatus:
        return EDLStatus(
            coordinate=data["coordinate"],
            sigma=data["sigma"],
            phi=data["phi"],
            efield=data.get("efield"),
            rho=data.get("rho"),
            ion_conc=data["ion_conc"],
            epsilon_r=data.get("epsilon_r"),
        )


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

    def serialize(self) -> Dict[str, unxt.Quantity | jax.Array | float]:
        return {
            "phi": self.phi.value.tolist(),
            "coverage": self.coverage.tolist(),
            "temperature": self.temperature.value.tolist(),
            "n_et": self.n_et,
            "coverage_max": self.coverage_max,
            "lateral_interaction": (
                self.lateral_interaction.value.tolist()
                if self.lateral_interaction is not None
                else None
            ),
        }

    def deserialize(
        data: Dict[str, unxt.Quantity | jax.Array | float],
    ) -> IsothermStatus:
        return IsothermStatus(
            phi=data["phi"],
            coverage=data["coverage"],
            temperature=data["temperature"],
            n_et=data["n_et"],
            coverage_max=data.get("coverage_max", 1.0),
            lateral_interaction=data.get("lateral_interaction"),
        )
