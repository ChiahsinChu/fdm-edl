# SPDX-License-Identifier: GPL-3.0-or-later
import copy
import pathlib

import jax.numpy as jnp

from fdm_edl.edl.electrode import Electrode
from fdm_edl.edl.electrolyte import Electrolyte
from fdm_edl.solver import Solver
from fdm_edl.utils import load_dict

# import matplotlib.pyplot as plt


class ElectricalDoubleLayer:
    def __init__(self, params: dict | str | pathlib.Path):
        if isinstance(params, dict):
            self._params = copy.deepcopy(params)
        elif isinstance(params, (str, pathlib.Path)):
            self._params = load_dict(params)
        else:
            raise ValueError("params must be a dict or a path to a yaml/json file")

        self.electrode_params = self._params.pop("electrode", {})
        self.electrolyte_params = self._params.pop("electrolyte", {})
        self.solver_params = self._params.pop("solver", {})

        self.electrode = Electrode(global_params=self._params, **self.electrode_params)
        self.electrolyte = Electrolyte(
            global_params=self._params, **self.electrolyte_params
        )

        self.solver = Solver(**self.solver_params)

    def set_solver(self, solver_params: dict):
        self.solver = Solver(**solver_params)

    def compute(self, x, phi_wall):
        """
        Solve for electrostatic potential profile with boundary conditions.
        """
        n_grid_int = len(x) - 2  # Number of interior points
        phi_int = jnp.zeros(n_grid_int)  # Initial guess

        result = self.solver.solve(self.get_residual, phi_int, phi_wall, 0.0, x)
        phi_int = result.x

        return jnp.concatenate([jnp.array([phi_wall]), phi_int, jnp.array([0.0])])

    def get_residual(
        self,
        phi_int,
        phi_left,
        phi_right,
        x,
    ):
        """
        Generalized Poisson-Boltzmann Residual
        """
        dx = (jnp.diff(x)[1:] + jnp.diff(x)[:-1]) / 2.0  # Distance between midpoints
        # get full phi including boundaries
        phi = jnp.concatenate([jnp.array([phi_left]), phi_int, jnp.array([phi_right])])

        # Laplace term: d2phi/dx2
        d2phi = (phi[:-2] - 2 * phi[1:-1] + phi[2:]) / dx**2

        # Generalized Boltzmann charge density term
        # rho = sum( q_j * c_j * exp(-q_j * phi) )
        rho_ion = 0.0
        for data in self.electrolyte.ions.values():
            q = data["charge"]
            c = data["conc"]
            rho_ion += q * c * jnp.exp(-q * phi[1:-1])

        # Residual = Laplacian + rho/epsilon (assuming epsilon is 1 for dimensionless)
        return d2phi + rho_ion
