# SPDX-License-Identifier: GPL-3.0-or-later
"""Physical constants used by fdm-edl."""

import unxt
from scipy import constants as _const

# elementary charge in Coulombs
ELEMENTARY_CHARGE = unxt.Quantity(_const.elementary_charge, "C")
# vacuum permittivity in F/m
VACUUM_PERMITTIVITY = unxt.Quantity(_const.epsilon_0, "F/m")
# Boltzmann constant in J/K
BOLTZMANN_CONSTANT = unxt.Quantity(_const.Boltzmann, "J/K")
# Avogadro's number in 1/mol
AVOGADRO_NUMBER = unxt.Quantity(_const.Avogadro, "1/mol")

# Derived constants
# Faraday constant in C/mol
FARADAY_CONSTANT = ELEMENTARY_CHARGE * AVOGADRO_NUMBER
# molar gas constant in J/(mol*K)
MOLAR_GAS_CONSTANT = BOLTZMANN_CONSTANT * AVOGADRO_NUMBER
