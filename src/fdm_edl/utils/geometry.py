# SPDX-License-Identifier: GPL-3.0-or-later
class Geometry:
    def __init__(self, dim: int = 1, **kwargs):
        """ """

        self.dim = dim
        # set all kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
