# SPDX-License-Identifier: GPL-3.0-or-later
"""Adsorption isotherm models for electrode surface coverage."""

from .base import BaseIsotherm
from .frumkin import FrumkinIsotherm
from .langmuir import LangmuirIsotherm

__all__ = [
    "BaseIsotherm",
    "LangmuirIsotherm",
    "FrumkinIsotherm",
]
