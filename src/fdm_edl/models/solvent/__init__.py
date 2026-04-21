# SPDX-License-Identifier: GPL-3.0-or-later
"""Solvent dielectric response models for field-dependent permittivity.

This package defines abstractions and concrete implementations used by the
electrolyte model to evaluate the relative permittivity as a function of the
local electric field.
"""

from .base import BaseSolvent

__all__ = ["BaseSolvent"]
