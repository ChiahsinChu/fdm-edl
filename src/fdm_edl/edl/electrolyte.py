# SPDX-License-Identifier: GPL-3.0-or-later
from dataclasses import dataclass
from typing import List

import quaxed.numpy as jnp
import unxt
from astropy.units import cds

from .. import _constants


@dataclass(frozen=True)
class Ion:
    """Ionic species definition for an electrolyte.

    Parameters
    ----------
    name : str
        Human-readable ion label (for example, ``"Na+"`` or ``"Cl-"``).
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
    radius: unxt.Quantity = unxt.Quantity(
        0.0, "angstrom"
    )  # Optional: ionic radius for steric effects


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

    ions: List[Ion]
    temperature: unxt.Quantity
    epsilon: unxt.Quantity
    electroneutrality: bool = True

    def __post_init__(self):
        """Validate electrolyte consistency after dataclass initialization.

        Raises
        ------
        ValueError
            If ``electroneutrality`` is enabled and the total charge
            concentration is nonzero.
        """

        if len(self.ions) > 0:
            tot_charge = unxt.Q(
                0.0, self.ions[0].charge.unit * self.ions[0].molar_conc.unit
            )
            for ion in self.ions:
                q = ion.charge
                c = ion.molar_conc
                tot_charge += q * c
            if self.electroneutrality and tot_charge != 0.0:
                raise ValueError(
                    f"Electroneutrality condition not satisfied. Total charge concentration: {tot_charge.value} {tot_charge.unit}"
                )

    @property
    def ionic_strength(self) -> unxt.Quantity:
        """
        Calculate the ionic strength of the solution.

        Returns
        -------
        unxt.Quantity
            Ionic strength, computed as
            :math:`I = \frac{1}{2}\sum_i z_i^2 c_i`.
        """
        ionic_strength = unxt.Q(0.0, "mol / L")
        if len(self.ions) > 0:
            for ion in self.ions:
                z = ion.charge.to(cds.e).value
                c = ion.molar_conc
                ionic_strength += 0.5 * z**2 * c
        return ionic_strength

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
            return unxt.Q(
                float("inf"), "angstrom"
            )  # Infinite Debye length for pure solvent

        debye_length = jnp.sqrt(
            self.epsilon
            * _constants.BOLTZMANN_CONSTANT
            * self.temperature
            / (
                2
                * _constants.AVOGADRO_NUMBER
                * _constants.ELEMENTARY_CHARGE**2
                * self.ionic_strength
            )
        )
        return debye_length.to("angstrom")  # Convert to Angstrom (Å)
