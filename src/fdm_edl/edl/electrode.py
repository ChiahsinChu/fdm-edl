# SPDX-License-Identifier: GPL-3.0-or-later
from dataclasses import dataclass

import unxt

from fdm_edl.utils.geometry import Geometry


@dataclass(frozen=True)
class Electrode:
    metal: str = "Pt"
    temperature: unxt.Quantity = None  # in K


class DeprecatedElectrode:
    """
    Electrode component of an EDL simulation.

    Parameters
    ----------
    global_params : dict
        Global simulation parameters (e.g., ``temperature``, ``epsilon_r``)
        that are set as instance attributes.
    **kwargs
        Additional keyword arguments.  The key ``geometry`` (dict) is
        extracted to construct a :class:`~fdm_edl.utils.geometry.Geometry`
        instance; all remaining keys are set as instance attributes.

    Attributes
    ----------
    geometry : Geometry
        Geometric description of the electrode domain.
    """

    def __init__(self, global_params: dict, **kwargs):
        """
        Initialize the Electrode.

        Parameters
        ----------
        global_params : dict
            Global simulation parameters set as instance attributes.
        **kwargs
            Keyword arguments; ``geometry`` (dict) builds a
            :class:`~fdm_edl.utils.geometry.Geometry` instance; remaining
            keys are set as attributes.
        """
        # set global params as attributes
        for key, value in global_params.items():
            setattr(self, key, value)
        _geometry = kwargs.pop("geometry", {})
        self.geometry = Geometry(**_geometry)

        # set all kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
