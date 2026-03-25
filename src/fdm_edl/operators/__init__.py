# SPDX-License-Identifier: GPL-3.0-or-later
"""Discrete differential operators for FDM-EDL."""

from .laplacian import LaplacianOperator, LaplacianOperator1D

__all__ = [
    "LaplacianOperator",
    "LaplacianOperator1D",
]
