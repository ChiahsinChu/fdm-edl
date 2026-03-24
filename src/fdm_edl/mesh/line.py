# SPDX-License-Identifier: GPL-3.0-or-later
"""1-D line mesh generation via Gmsh."""

from typing import List

import numpy as np

from fdm_edl.mesh.base import BaseMesh


class LineMesh(BaseMesh):
    """Mesh for a 1-D line segment with variable node spacing.

    Inherits from :class:`~fdm_edl.mesh.base.BaseMesh` and provides a
    concrete :meth:`define_geometry` that creates a single line between
    two points using the OpenCASCADE kernel.

    Examples
    --------
    >>> mesh = LineMesh()
    >>> coords = mesh.run(xleft=0.0, xright=1.0,
    ...                   mesh_size_left=0.01, mesh_size_right=0.05)
    >>> coords.shape  # (N, 3)
    """

    def define_geometry(
        self,
        points: List[float] | np.ndarray,
        mesh_sizes: List[float] | np.ndarray | float,
    ):
        """Define a line segment from *xleft* to *xright*.

        Parameters
        ----------
        points:
            List of x-coordinates for the points.
        mesh_sizes:
            List of mesh sizes for each point.
        """
        occ = self.model.occ

        if isinstance(mesh_sizes, float):
            mesh_sizes = [mesh_sizes] * len(points)
        elif len(mesh_sizes) != len(points):
            raise ValueError(
                "mesh_sizes must be a float or have the same length as points."
            )

        pts = [occ.addPoint(x, 0, 0, h) for x, h in zip(points, mesh_sizes)]
        _lines = [occ.addLine(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]

        # ask gmsh to synchronize the CAD kernel with the Gmsh model before meshing
        occ.synchronize()
