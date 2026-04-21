# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import quaxed.numpy as jnp
import unxt
from astropy.units import cds  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from typing import Dict

from ..models.solvent.base import BaseSolvent
from ..utils import constants
from ..utils import unit_conversion as uc


@dataclass(frozen=True)
class Ion:
    """Ionic species definition for an electrolyte.

    Parameters
    ----------
    name : str
        Human-readable ion label (for example, ``"Na"`` or ``"Cl"``).
    charge : unxt.Quantity
        Ionic charge as a quantity (typically in multiples of ``e``).
    molar_conc : unxt.Quantity
        Bulk molar concentration of the ion (for example, ``mol / L``).
    radius : unxt.Quantity, optional
        Effective ionic radius used by steric models. Defaults to
        ``0.0 angstrom``.
    """

    name: str
    charge: unxt.Quantity
    molar_conc: unxt.Quantity
    radius: unxt.Quantity = unxt.Quantity(0.0, "angstrom")
    _charge: float = field(init=False, repr=False)
    _molar_conc: float = field(init=False, repr=False)
    _radius: float = field(init=False, repr=False)
    molar_volume: unxt.Quantity = field(init=False, repr=False)
    _molar_volume: float = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "_charge",
            self.charge.to(uc.UNIT_SYSTEMS["metal"]["electrical charge"]).value,
        )
        object.__setattr__(
            self,
            "_molar_conc",
            self.molar_conc.to(
                unxt.unit("mol") / uc.UNIT_SYSTEMS["metal"]["volume"]
            ).value,
        )
        object.__setattr__(
            self, "_radius", self.radius.to(uc.UNIT_SYSTEMS["metal"]["length"]).value
        )
        molar_volume = ((2 * self.radius) ** 3 * constants.AVOGADRO_NUMBER).to(
            "L / mol"
        )
        object.__setattr__(
            self,
            "molar_volume",
            molar_volume,
        )
        object.__setattr__(
            self,
            "_molar_volume",
            molar_volume.to(
                uc.UNIT_SYSTEMS["metal"]["volume"] / unxt.unit("mol")
            ).value,
        )


@dataclass(frozen=True)
class Electrolyte:
    """Electrolyte solution model used by the EDL solver.

    Parameters
    ----------
    ions : list[Ion]
        Collection of ionic species present in solution.
    temperature : unxt.Quantity
        Absolute temperature.
    epsilon : unxt.Quantity
        Permittivity of the medium.
    electroneutrality : bool, optional
        If ``True``, enforce global electroneutrality at initialization.
    """

    ions: Dict[str, Ion]
    solvent: BaseSolvent
    temperature: unxt.Quantity
    # epsilon: unxt.Quantity
    # epsilon_r: float = 78.5
    electroneutrality: bool = True
    _temperature: float = field(init=False, repr=False)
    # _epsilon: float = field(init=False, repr=False)

    def __post_init__(self):
        """Validate electrolyte consistency after dataclass initialization.

        Raises
        ------
        ValueError
            If ``electroneutrality`` is enabled and the total charge
            concentration is nonzero.
        """

        if len(self.ions) > 0:
            tot_charge = 0.0
            for ion in self.ions.values():
                q = ion.charge.value
                c = ion.molar_conc.value
                tot_charge += q * c
            if self.electroneutrality and tot_charge != 0.0:
                raise ValueError(
                    f"Electroneutrality condition not satisfied. Total charge concentration: {tot_charge} {self.ions[0].charge.unit * self.ions[0].molar_conc.unit}"
                )

        object.__setattr__(
            self,
            "_temperature",
            self.temperature.to(uc.UNIT_SYSTEMS["metal"]["temperature"]).value,
        )

    @property
    def ionic_strength(self) -> unxt.Quantity:
        """
        Calculate the ionic strength of the solution.

        Returns
        -------
        unxt.Quantity
            Ionic strength, computed as
            :math:`I = \frac{1}{2}\\sum_i z_i^2 c_i`.
        """
        ionic_strength = unxt.Quantity(0.0, "mol / L")
        if len(self.ions) > 0:
            for ion in self.ions.values():
                z = ion.charge.to(cds.e).value
                c = ion.molar_conc
                ionic_strength += 0.5 * z**2 * c
        return cast(unxt.Quantity, ionic_strength.to("mol / L"))

    @property
    def debye_length(self) -> unxt.Quantity:
        """
        Calculate the Debye length based on the ionic strength of the solution.

        Returns
        -------
        unxt.Quantity
            Debye length of the electrolyte. Returns ``inf angstrom`` for
            zero ionic strength.
        """
        if self.ionic_strength == 0.0:
            return unxt.Quantity(
                float("inf"), "angstrom"
            )  # Infinite Debye length for pure solvent

        debye_length = jnp.sqrt(
            self.solvent.eps_0
            * constants.VACUUM_PERMITTIVITY
            * constants.BOLTZMANN_CONSTANT
            * self.temperature
            / (
                2
                * constants.AVOGADRO_NUMBER
                * constants.ELEMENTARY_CHARGE**2
                * self.ionic_strength
            )
        )
        return cast(
            unxt.Quantity,
            debye_length.to(uc.UNIT_SYSTEMS["metal"]["length"]),
        )
