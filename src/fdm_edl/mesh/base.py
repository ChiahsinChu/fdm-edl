# SPDX-License-Identifier: GPL-3.0-or-later
"""Abstract base class for Gmsh-backed mesh objects."""

import gmsh
import numpy as np

"""
todo list:
- [ ] add connectivity to class attribute

"""


class BaseMesh:
    """Base class for mesh generation using the Gmsh library.

    Subclasses must implement :meth:`define_geometry` to describe the
    geometric entities that Gmsh should mesh.

    Parameters
    ----------
    dim:
        Spatial dimension of the mesh (default: 1).
    """

    def __init__(self, dim: int = 1):
        self.dim = dim

        gmsh.initialize()
        gmsh.model.add("workspace")
        self.model = gmsh.model

    def __del__(self):
        gmsh.finalize()

    def run(self, **kwargs) -> np.ndarray:
        """Generate the mesh and return node coordinates.

        Calls :meth:`define_geometry` with any provided keyword arguments,
        triggers Gmsh mesh generation, and returns the resulting node
        positions as a ``(N, 3)`` NumPy array (x, y, z columns).

        Parameters
        ----------
        **kwargs:
            Forwarded verbatim to :meth:`define_geometry`.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(N, 3)`` containing the coordinates of all
            mesh nodes.
        """
        self.define_geometry(**kwargs)
        # Generate Mesh
        gmsh.model.mesh.generate(self.dim)
        # Get all node tags and their coordinates
        _node_tags, _meshes, _ = gmsh.model.mesh.getNodes()
        # Reshape coords: Gmsh gives [x1, y1, z1, x2, y2, z2...]
        # We turn it into a (N, 3) array
        return _meshes.reshape(-1, 3)[:, : self.dim]

    def define_geometry(self, **kwargs):
        """Define Gmsh geometric entities for meshing.

        Must be overridden by subclasses.  All keyword arguments come
        directly from :meth:`run`.

        Raises
        ------
        NotImplementedError
            Always, unless overridden by a subclass.
        """
        raise NotImplementedError("Subclasses must implement define_geometry()")
