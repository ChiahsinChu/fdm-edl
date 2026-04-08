# SPDX-License-Identifier: GPL-3.0-or-later
from dataclasses import dataclass
from typing import Dict

import unxt


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
