# SPDX-License-Identifier: GPL-3.0-or-later
"""Physical constants used by fdm-edl."""

from scipy.constants import Avogadro, Boltzmann, elementary_charge, epsilon_0

# units: SI
ELEMENTARY_CHARGE = float(elementary_charge)
EPSILON_0 = float(epsilon_0)
BOLTZMANN = float(Boltzmann)
AVOGADRO = float(Avogadro)

__all__ = ["ELEMENTARY_CHARGE", "EPSILON_0", "BOLTZMANN", "AVOGADRO"]
