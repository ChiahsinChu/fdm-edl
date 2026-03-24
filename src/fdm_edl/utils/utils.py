# SPDX-License-Identifier: GPL-3.0-or-later
import unxt
from astropy.units import cds


def to_unxtq(params: dict):
    """Convert a parameter mapping into a ``unxt.Quantity``.

    Parameters
    ----------
    params : dict
        Mapping containing quantity constructor fields accepted by
        :meth:`unxt.Quantity.from_`. The ``"unit"`` entry may be the
        string ``"e"`` for elementary charge.

    Returns
    -------
    unxt.Quantity
        Quantity object constructed from ``params``.
    """

    # Find full list for available units: https://docs.astropy.org/en/stable/units/ref_api.html
    # elementary charge
    if params["unit"] == "e":
        params["unit"] = cds.e
    return unxt.Quantity.from_(params)
