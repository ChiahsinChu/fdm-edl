# SPDX-License-Identifier: GPL-3.0-or-later
from fdm_edl.utils.geometry import Geometry


class Electrode:
    def __init__(self, global_params: dict, **kwargs):
        # set global params as attributes
        for key, value in global_params.items():
            setattr(self, key, value)
        _geometry = kwargs.pop("geometry", {})
        self.geometry = Geometry(**_geometry)

        # set all kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
