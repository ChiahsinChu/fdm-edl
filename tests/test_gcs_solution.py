# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the Gouy-Chapman-Stern model.

Compares the FDM numerical solution (using SternLayerBC at the OHP)
against the analytical GCS solution for a symmetric 1:1 electrolyte.
"""

import unittest
from pathlib import Path

import numpy as np
import quaxed.numpy as jnp
import unxt

from fdm_edl.api import ElectricalDoubleLayer
from fdm_edl.bc import DirichletBC, SternLayerBC
from fdm_edl.test import GCSPoissonBoltzmann
from fdm_edl.utils import constants


class TestGCSModel(unittest.TestCase):
    """Test Gouy-Chapman-Stern numerical vs analytical."""

    def setUp(self) -> None:
        example_input = Path(__file__).resolve().parent / "data" / "NaCl_gcs.json"
        self.edl_obj = ElectricalDoubleLayer(example_input)

    def _run_gcs_comparison(self, phi0_V: float, stern_d_nm: float, stern_eps_r: float):
        """Helper: run numerical GCS and compare to analytical."""
        n_grid = 500
        x = unxt.Quantity(jnp.linspace(0.0, 50.0, n_grid), unit="nm")

        electrode_potential = unxt.Quantity(phi0_V, "V")
        stern_thickness = unxt.Quantity(stern_d_nm, "nm")
        stern_permittivity = stern_eps_r * constants.VACUUM_PERMITTIVITY
        bulk_permittivity = self.edl_obj.electrolyte.epsilon

        # Analytical reference
        gcs = GCSPoissonBoltzmann(
            edl_obj=self.edl_obj,
            x=x,
            electrode_potential=electrode_potential,
            stern_thickness=stern_thickness,
            stern_permittivity=stern_permittivity,
        )

        # Numerical solution with SternLayerBC at OHP (node 0)
        boundary_conditions = [
            SternLayerBC(
                node_indices=[0],
                electrode_potential=electrode_potential,
                stern_thickness=stern_thickness,
                stern_permittivity=stern_permittivity,
                bulk_permittivity=bulk_permittivity,
                neighbor_indices=[1],
            ),
            DirichletBC(
                [n_grid - 1],
                unxt.Quantity(jnp.zeros((1,)), unit="V"),
            ),
        ]
        self.edl_obj.compute(x, boundary_conditions)

        np.testing.assert_allclose(
            gcs.phi.value,
            self.edl_obj.result.solution.value,
            atol=1e-4,
            rtol=1e-4,
        )
        return gcs

    def test_gcs_moderate_potential(self) -> None:
        """GCS with moderate electrode potential (0.1 V)."""
        self._run_gcs_comparison(phi0_V=0.1, stern_d_nm=0.3, stern_eps_r=10.0)

    def test_gcs_high_potential(self) -> None:
        """GCS with higher electrode potential (0.25 V)."""
        self._run_gcs_comparison(phi0_V=0.25, stern_d_nm=0.3, stern_eps_r=10.0)

    def test_gcs_thick_stern_layer(self) -> None:
        """GCS with a thicker Stern layer (0.5 nm)."""
        self._run_gcs_comparison(phi0_V=0.1, stern_d_nm=0.5, stern_eps_r=6.0)

    def test_gcs_phi_d_less_than_phi0(self) -> None:
        """The OHP potential φ_d should always be smaller than φ₀."""
        gcs = self._run_gcs_comparison(phi0_V=0.2, stern_d_nm=0.3, stern_eps_r=10.0)
        self.assertLess(abs(gcs.phi_d.to("V").value), abs(0.2))

    def test_gcs_negative_potential(self) -> None:
        """GCS with negative electrode potential."""
        self._run_gcs_comparison(phi0_V=-0.15, stern_d_nm=0.3, stern_eps_r=10.0)
