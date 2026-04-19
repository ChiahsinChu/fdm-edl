# SPDX-License-Identifier: GPL-3.0-or-later
from dataclasses import dataclass

import unxt


@dataclass(frozen=True)
class Electrode:
    """Electrode component of an EDL simulation.

    Parameters
    ----------
    metal : str, optional
        Metal element symbol (default: ``"Pt"``).
    temperature : unxt.Quantity or None, optional
        Absolute temperature (default: ``None``).
    """

    metal: str = "Pt"
    temperature: unxt.Quantity | None = None  # in K
