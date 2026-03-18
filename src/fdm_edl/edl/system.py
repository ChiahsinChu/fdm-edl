# SPDX-License-Identifier: GPL-3.0-or-later
import copy
import pathlib

import jax.numpy as jnp
from matplotlib import pyplot as plt

from fdm_edl.edl.electrode import Electrode
from fdm_edl.edl.electrolyte import Electrolyte
from fdm_edl.solver import Solver
from fdm_edl.utils import AVOGADRO, BOLTZMANN, ELEMENTARY_CHARGE, EPSILON_0, load_dict


class ElectricalDoubleLayer:
    """
    1-D Electrical Double Layer (EDL) model based on the Poisson-Boltzmann
    equation, solved with a finite-difference method.

    Parameters
    ----------
    params : dict or str or pathlib.Path
        Model parameters as a dictionary or as a path to a JSON/YAML file.
        Required keys:

        ``temperature`` : float
            Absolute temperature in Kelvin.

        Optional keys:

        ``epsilon_r`` : float
            Relative permittivity of the solvent (default: 78.5).
        ``electrode`` : dict
            Keyword arguments forwarded to :class:`~fdm_edl.edl.Electrode`.
        ``electrolyte`` : dict
            Keyword arguments forwarded to
            :class:`~fdm_edl.edl.Electrolyte`.
        ``solver`` : dict
            Keyword arguments forwarded to :class:`~fdm_edl.solver.Solver`.

    Attributes
    ----------
    temperature : float
        System temperature in Kelvin.
    epsilon_r : float
        Relative permittivity of the solvent.
    epsilon : float
        Absolute permittivity (``epsilon_r * EPSILON_0``) in F/m.
    beta : float
        Inverse thermal voltage (``e / (k_B T)``) in 1/V.
    faraday : float
        Faraday constant (``e * N_A``) in C/mol.
    gas_constant : float
        Molar gas constant (``k_B * N_A``) in J/(mol·K).
    electrode : Electrode
        Electrode component of the EDL model.
    electrolyte : Electrolyte
        Electrolyte component of the EDL model.
    solver : Solver
        Nonlinear root-finding solver.
    coordinates : ndarray or None
        Grid coordinates (nm) from the last :meth:`compute` call.
    solution : jax.Array or None
        Full electrostatic potential profile (V) from the last
        :meth:`compute` call.
    results : RootSolveResult or None
        Full solver result from the last :meth:`compute` call.

    Raises
    ------
    ValueError
        If ``params`` is not a dict or a valid path, or if ``temperature``
        is missing from the parameter dictionary.
    """

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

        if "temperature" not in self._params:
            raise ValueError("temperature (K) is required in global params for SI mode")
        self.temperature = float(self._params["temperature"])

        # Relative permittivity defaults to water at room temperature.
        self.epsilon_r = float(
            self._params.get("epsilon_r", self._params.get("epsilon_r", 78.5))
        )
        self.epsilon = self.epsilon_r * EPSILON_0
        self.beta = ELEMENTARY_CHARGE / (BOLTZMANN * self.temperature)
        self.faraday = ELEMENTARY_CHARGE * AVOGADRO
        self.gas_constant = BOLTZMANN * AVOGADRO

        self.electrode = Electrode(global_params=self._params, **self.electrode_params)
        self.electrolyte = Electrolyte(
            global_params=self._params, **self.electrolyte_params
        )

        self.solver = Solver(**self.solver_params)
        # placeholders for results
        self.coordinates = None
        self.results = None

    def set_solver(self, solver_params: dict):
        """
        Set up the nonlinear solver with new parameters.

        Parameters
        ----------
        solver_params : dict
            Keyword arguments forwarded to :class:`~fdm_edl.solver.Solver`.
        """
        self.solver = Solver(**solver_params)

    def compute(self, coordinates, phi_wall):
        """
        Solve for the electrostatic potential profile.

        Uses the Poisson-Boltzmann equation with Dirichlet boundary
        conditions: ``phi(x=0) = phi_wall`` and ``phi(x=L) = 0``.

        Parameters
        ----------
        coordinates : array-like of float
            1-D grid coordinates in nanometres, including both boundary
            nodes.
        phi_wall : float
            Electrostatic potential at the electrode surface in Volts.

        Returns
        -------
        solution : jax.Array
            Electrostatic potential (V) at every grid node, including the
            two boundary values.
        """
        n_grid_int = len(coordinates) - 2  # Number of interior points
        phi_int = jnp.zeros(n_grid_int)  # Initial guess

        result = self.solver.solve(
            self.get_residual, phi_int, phi_wall, 0.0, coordinates
        )
        self.coordinates = coordinates
        self.results = result
        self.solution = jnp.concatenate(
            [jnp.array([phi_wall]), result.solution, jnp.array([0.0])]
        )

        return self.solution

    def get_residual(
        self,
        phi_int,
        phi_left,
        phi_right,
        coordinates,
    ):
        """
        Compute the Poisson-Boltzmann residual at interior grid nodes.

        Parameters
        ----------
        phi_int : jax.Array, shape (N,)
            Electrostatic potential (V) at the *N* interior grid nodes.
        phi_left : float
            Dirichlet value at the left boundary (electrode surface) in V.
        phi_right : float
            Dirichlet value at the right boundary (bulk) in V.
        coordinates : array-like of float, shape (N+2,)
            Grid node positions in nanometres, including boundary nodes.

        Returns
        -------
        residual : jax.Array, shape (N,)
            Residual of the discretised Poisson equation at each interior
            node.  Zero when the solution satisfies the Poisson-Boltzmann
            equation.

        Notes
        -----
        The Poisson-Boltzmann equation in SI units is

        .. math::

            \\frac{d^2\\phi}{dx^2}
            + \\frac{\\rho_{\\mathrm{ion}}}{\\varepsilon} = 0

        where the ionic charge density is

        .. math::

            \\rho_{\\mathrm{ion}} = F \\sum_j z_j c_j^0
                \\exp\\!\\left(-\\frac{z_j F \\phi}{RT}\\right).

        The second derivative is approximated with a central
        finite-difference stencil.  Input coordinates in nm are converted
        to metres internally.
        """
        # Coordinates are provided in nm, while SI Poisson-Boltzmann is in meters.
        dx = (jnp.diff(coordinates)[1:] + jnp.diff(coordinates)[:-1]) / 2.0 * 1e-9
        # get full phi including boundaries
        phi = jnp.concatenate([jnp.array([phi_left]), phi_int, jnp.array([phi_right])])

        # Laplace term: d2phi/dx2
        d2phi = (phi[:-2] - 2 * phi[1:-1] + phi[2:]) / dx**2

        # Generalized Boltzmann charge density term
        # rho = sum( q_j * c_j * exp(-q_j * phi) )
        rho_ion = 0.0
        for data in self.electrolyte.ions.values():
            z = data["charge"]
            c_molm3 = data["conc"] * 1000.0  # input conc is mol/L
            rho_ion += (
                self.faraday
                * z
                * c_molm3
                * jnp.exp(
                    -z
                    * self.faraday
                    * phi[1:-1]
                    / (self.gas_constant * self.temperature)
                )
            )

        # SI Poisson equation: d2phi/dx2 + rho/epsilon = 0
        return d2phi + rho_ion / self.epsilon

    def plot(self):
        """
        Plot the electrostatic potential and ion concentration profiles.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing both subplots.
        axs : ndarray of matplotlib.axes.Axes, shape (2,)
            Array of axes: ``axs[0]`` holds the potential profile and
            ``axs[1]`` the ion concentration profiles.

        Notes
        -----
        :meth:`compute` must be called before :meth:`plot`.
        """
        nrows = 2
        ncols = 1
        fig, axs = plt.subplots(
            nrows, ncols, figsize=(nrows * 4, ncols * 5), sharex=True
        )

        # Plot Potential
        ax = axs[0]
        ax.plot(self.coordinates, self.solution, color="black", lw=2)
        ax.set_title("Electrostatic Potential $\phi(x)$")
        ax.set_ylabel("$\phi$ (V)")
        ax.grid(True)

        # Plot Concentrations
        ax = axs[1]
        total_q_conc = 0.0
        for name, data in self.electrolyte.ions.items():
            z = data["charge"]
            c0 = data["conc"]
            # Local concentration in mol/L.
            local_c = c0 * jnp.exp(
                -z
                * self.faraday
                * self.solution
                / (self.gas_constant * self.temperature)
            )
            ax.plot(self.coordinates, local_c, label=f"{name} ($z={z}$)")
            total_q_conc += local_c * z

        ax.plot(
            self.coordinates,
            total_q_conc,
            label="Total Charge Concentration",
            color="red",
            lw=2,
            linestyle="--",
        )

        ax.set_title("Ion Concentrations")
        ax.set_ylabel("Concentration (M)")
        ax.legend()
        ax.grid(True)

        ax.set_xlabel("nm")
        plt.tight_layout()
        return fig, axs
