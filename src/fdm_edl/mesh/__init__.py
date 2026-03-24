# SPDX-License-Identifier: GPL-3.0-or-later
"""Mesh generation utilities for finite-difference EDL simulations.

Provides structured 1-D (and extensible N-D) mesh classes built on top of
Gmsh.
"""

from .line import LineMesh

__all__ = [
    "LineMesh",
]
