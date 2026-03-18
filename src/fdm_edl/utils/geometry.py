# SPDX-License-Identifier: GPL-3.0-or-later
class Geometry:
    """
    Lightweight descriptor for a simulation domain geometry.

    Parameters
    ----------
    dim : int, optional
        Spatial dimensionality of the domain (default: 1).
    **kwargs
        Additional geometric parameters (e.g., ``length``, ``radius``)
        that are set as instance attributes.

    Attributes
    ----------
    dim : int
        Spatial dimensionality.
    """

    def __init__(self, dim: int = 1, **kwargs):
        """
        Initialize the Geometry.

        Parameters
        ----------
        dim : int, optional
            Spatial dimensionality (default: 1).
        **kwargs
            Additional parameters set as instance attributes.
        """

        self.dim = dim
        # set all kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
