# SPDX-License-Identifier: GPL-3.0-or-later
"""Unit conversion utilities for FDM-EDL.

Follows the `LAMMPS unit convention <https://docs.lammps.org/units.html>`_.
Supports ``"metal"``, ``"real"``, and ``"si"`` unit systems.
The **internal** unit system used for computation is ``"metal"``.

Quick start
-----------
>>> from fdm_edl.utils.unit_conversion import get_conversion_factor
>>> get_conversion_factor("length", "si")       # 1 m -> metal (Å)
1e+10
>>> get_conversion_factor("energy", "metal", "si")  # 1 eV -> SI (J)
1.602176634e-19

The 7 SI base dimensions are defined together with commonly used derived
dimensions (energy, charge, force, …).  Conversion factors are computed
once at import time from :mod:`unxt` and :mod:`astropy.units`.
"""

from __future__ import annotations

from typing import Literal, cast

import astropy.units as apyu  # type: ignore[import-untyped]
import unxt
from astropy.units import cds  # type: ignore[import-untyped]

from .constants import AVOGADRO_NUMBER, ELEMENTARY_CHARGE

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
UnitSystem = Literal["metal", "real", "si"]

# ---------------------------------------------------------------------------
# Human-readable unit labels  (display / documentation)
# ---------------------------------------------------------------------------
unit_labels: dict[str, dict[str, str]] = {
    "metal": {
        # 7 SI base dimensions
        "length": "Å",
        "time": "ps",
        "mass": "g/mol",
        "current": "A",
        "temperature": "K",
        "amount": "mol",
        "luminous_intensity": "cd",
        # derived
        "energy": "eV",
        "charge": "e",
        "force": "eV/Å",
        "velocity": "Å/ps",
        "pressure": "bar",
        "electric_potential": "V",
        "electric_field": "V/Å",
        "dipole": "e·Å",
        "permittivity": "e²/(eV·Å)",
        "number_density": "1/Å³",
        "charge_density": "e/Å³",
        "surface_charge_density": "e/Å²",
    },
    "real": {
        "length": "Å",
        "time": "fs",
        "mass": "g/mol",
        "current": "A",
        "temperature": "K",
        "amount": "mol",
        "luminous_intensity": "cd",
        "energy": "kcal/mol",
        "charge": "e",
        "force": "kcal/(mol·Å)",
        "velocity": "Å/fs",
        "pressure": "atm",
        "electric_potential": "V",
        "electric_field": "V/Å",
        "dipole": "e·Å",
        "permittivity": "e²·mol/(kcal·Å)",
        "number_density": "1/Å³",
        "charge_density": "e/Å³",
        "surface_charge_density": "e/Å²",
    },
    "si": {
        "length": "m",
        "time": "s",
        "mass": "kg",
        "current": "A",
        "temperature": "K",
        "amount": "mol",
        "luminous_intensity": "cd",
        "energy": "J",
        "charge": "C",
        "force": "N",
        "velocity": "m/s",
        "pressure": "Pa",
        "electric_potential": "V",
        "electric_field": "V/m",
        "dipole": "C·m",
        "permittivity": "F/m",
        "number_density": "1/m³",
        "charge_density": "C/m³",
        "surface_charge_density": "C/m²",
    },
}

# ---------------------------------------------------------------------------
# unxt / astropy unit objects
# ---------------------------------------------------------------------------
base_unit_systems: dict[str, dict[str, apyu.UnitBase]] = {
    "metal": {
        "length": unxt.unit("angstrom"),
        "time": unxt.unit("ps"),
        "mass": unxt.unit("g/mol"),
        "temperature": unxt.unit("K"),
        "energy": unxt.unit("eV"),
        "electrical charge": cds.e,
        # "amount of substance": unxt.unit("mol"),
        "molar concentration": unxt.unit("mol/L"),
    },
    "real": {
        "length": unxt.unit("angstrom"),
        "time": unxt.unit("fs"),
        "mass": unxt.unit("g/mol"),
        "temperature": unxt.unit("K"),
        "energy": apyu.imperial.kcal / unxt.unit("mol"),
        "electrical charge": cds.e,
        # "amount of substance": unxt.unit("mol"),
        "molar concentration": unxt.unit("mol/L"),
    },
    "si": {
        "length": unxt.unit("m"),
        "time": unxt.unit("s"),
        "mass": unxt.unit("kg"),
        "temperature": unxt.unit("K"),
        "energy": unxt.unit("J"),
        "electrical charge": unxt.unit("C"),
        # "amount of substance": unxt.unit("mol"),
        "molar concentration": unxt.unit("mol/L"),
    },
}

_derived_formulas: dict[str, dict[str, int]] = {
    "electrical current": {"electrical charge": 1, "time": -1},
    "area": {"length": 2},
    "volume": {"length": 3},
    "force": {"energy": 1, "length": -1},
    "velocity": {"length": 1, "time": -1},
    "speed": {"length": 1, "time": -1},
    "electrical potential": {"energy": 1, "electrical charge": -1},
    "electrical field strength": {"energy": 1, "electrical charge": -1, "length": -1},
    "electrical dipole moment": {"electrical charge": 1, "length": 1},
    "surface charge density": {"electrical charge": 1, "length": -2},
    "polarization density": {"electrical charge": 1, "length": -2},
    "electrical flux density": {"electrical charge": 1, "length": -2},
    "electrical charge density": {"electrical charge": 1, "length": -3},
    "electrical capacitance": {"electrical charge": 2, "energy": -1},
    "permittivity": {"electrical charge": 2, "energy": -1, "length": -1},
    "pressure": {"energy": 1, "length": -3},
    "stress": {"energy": 1, "length": -3},
    "energy density": {"energy": 1, "length": -3},
}


def _build_derived_unit_dict(
    base_unit_dict: dict[str, apyu.UnitBase],
) -> dict[str, apyu.UnitBase]:
    """Compute full factor table (base + derived) from base factors."""
    derived_unit_dict = {}
    for kw, formula in _derived_formulas.items():
        u = unxt.unit("")
        for basic_unit, exp in formula.items():
            u *= base_unit_dict[basic_unit] ** exp
        derived_unit_dict[kw] = u
    return derived_unit_dict


UNIT_SYSTEMS = {
    kw: {**unit_dict, **_build_derived_unit_dict(unit_dict)}
    for kw, unit_dict in base_unit_systems.items()
}


def get_conversion_factor(
    data_type: str,
    unit_in: UnitSystem,
    unit_out: UnitSystem = "metal",
) -> float:
    """Return the multiplicative factor to convert *data_type* from *unit_in* to *unit_out*.

    Parameters
    ----------
    data_type : str
        Physical quantity name, e.g. ``"length"``, ``"energy"``,
        ``"electrical potential"``.  Must be a key in :data:`UNIT_SYSTEMS`.
    unit_in : {"metal", "real", "si"}
        Source unit system.
    unit_out : {"metal", "real", "si"}, default ``"metal"``
        Target unit system.

    Returns
    -------
    float
        Conversion factor such that
        ``value_in * get_conversion_factor(data_type, unit_in, unit_out)``
        gives the value in *unit_out*.

    Examples
    --------
    >>> get_conversion_factor("length", "si")  # 1 m -> Å
    1e+10
    """
    factor = (UNIT_SYSTEMS[unit_in][data_type] / UNIT_SYSTEMS[unit_out][data_type]).to(
        ""
    )
    return factor


def check_data_type(a: unxt.Quantity, data_type: str) -> None:
    """Verify that *a* has the expected physical dimension.

    Parameters
    ----------
    a : unxt.Quantity
        Quantity whose dimension is checked.
    data_type : str
        Expected physical type string understood by
        :func:`astropy.units.get_physical_type`.

    Raises
    ------
    ValueError
        If the dimension of *a* does not match *data_type*.
    """
    target_data_type = apyu.get_physical_type(data_type)
    if unxt.dimension_of(a) is not target_data_type:
        raise ValueError(
            f"Expected data_type {target_data_type}, but got {unxt.dimension_of(a)}"
        )


def rho_to_molar_concentration(rho: unxt.Quantity) -> unxt.Quantity:
    """Convert charge density to molar concentration (mol/L)."""
    check_data_type(rho, "electrical charge density")
    molar_conc = rho / (ELEMENTARY_CHARGE * AVOGADRO_NUMBER)
    return cast(unxt.Quantity, molar_conc.to("mol / L"))


__all__ = [
    "get_conversion_factor",
    "check_data_type",
    "rho_to_molar_concentration",
    "UNIT_SYSTEMS",
]
