# SPDX-License-Identifier: GPL-3.0-or-later
import copy
import pathlib

import quaxed.numpy as jnp
import unxt
from matplotlib import pyplot as plt

from fdm_edl.edl.electrode import Electrode
from fdm_edl.edl.electrolyte import Electrolyte, Ion
from fdm_edl.solver.base import BaseSolver
from fdm_edl.utils import load_dict, to_unxtq

from .. import _constants


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
            Keyword arguments forwarded to :class:`~fdm_edl.solver.BaseSolver`.

    Attributes
    ----------
    temperature : unxt.Quantity
        System temperature.
    dim : int
        Spatial dimension of the model (default: 1).
    electrode : Electrode
        Electrode component of the EDL model.
    electrolyte : Electrolyte
        Electrolyte component of the EDL model.
    solver : BaseSolver
        Nonlinear root-finding solver.
    coordinates : unxt.Quantity or None
        Grid coordinates from the last :meth:`compute` call, or ``None``
        before :meth:`compute` is called.
    result : RootSolveResult or None
        Full solver result from the last :meth:`compute` call, or ``None``
        before :meth:`compute` is called.  The assembled potential profile
        is available as ``result.solution`` and the grid coordinates as
        ``result.coordinates``.

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

        # get mandatory global parameters and set as attributes
        try:
            self.temperature = to_unxtq(self._params["temperature"])
        except KeyError:
            raise ValueError("`temperature` is required.")
        self.dim: int = self._params.pop("dim", 1)

        self.set_electrode(self._params.get("electrode", {}))
        self.set_electrolyte(self._params.get("electrolyte", {}))
        self.set_solver(self._params.get("solver", {}))

        # placeholders for results
        self.coordinates = None
        self.results = None

    def set_electrode(self, _params: dict):
        """
        Set up the electrode with new parameters.

        Parameters
        ----------
        _params : dict
            Keyword arguments forwarded to :class:`~fdm_edl.edl.Electrode`.
            Currently unused; the electrode is initialised with the system
            temperature only.
        """
        params = {
            "temperature": self.temperature,
        }
        self.electrode = Electrode(**params)

    def set_electrolyte(self, _params: dict):
        """
        Set up the electrolyte with new parameters.

        Parameters
        ----------
        _params : dict
            Keyword arguments forwarded to :class:`~fdm_edl.edl.Electrolyte`.
            Recognised keys:

            ``ions`` : dict
                Mapping of ion name to a sub-dict with keys ``charge``
                (ionic charge as a quantity string, e.g. ``"1 e"``) and
                ``molar_conc`` (bulk molar concentration as a quantity
                string, e.g. ``"0.1 mol/L"``).
            ``epsilon_r`` : float, optional
                Relative permittivity of the solvent (default: 78.5).
        """
        params = {
            "ions": [
                Ion(
                    name=name,
                    charge=to_unxtq(data["charge"]),
                    molar_conc=to_unxtq(data["molar_conc"]),
                )
                for name, data in _params.get("ions", {}).items()
            ],
            "epsilon": _params.get("epsilon_r", 78.5) * _constants.VACUUM_PERMITTIVITY,
            "temperature": self.temperature,
        }
        self.electrolyte = Electrolyte(**params)

    def set_solver(self, _params: dict):
        """
        Set up the nonlinear solver with new parameters.

        Parameters
        ----------
        _params : dict
            Keyword arguments forwarded to :class:`~fdm_edl.solver.BaseSolver`.
        """
        self.solver = BaseSolver(**_params)

    def compute(self, coordinates: unxt.Quantity, phi_wall: unxt.Quantity):
        """
        Solve for the electrostatic potential profile.

        Uses the Poisson-Boltzmann equation with Dirichlet boundary
        conditions: ``phi(x=0) = phi_wall`` and ``phi(x=L) = 0``.

        Parameters
        ----------
        coordinates : unxt.Quantity
            1-D grid coordinates including both boundary nodes.
        phi_wall : unxt.Quantity
            Electrostatic potential at the electrode surface.

        Returns
        -------
        None
            The result is stored in ``self.result``.  The assembled
            potential profile (including boundary values) is available as
            ``self.result.solution``; the grid coordinates are available
            as ``self.result.coordinates``.
        """
        # check coordinates shape is (N+2, dim) for some N
        # if coordinates.shape[1] != self.dim:
        #     raise ValueError("coordinates shape is incompatible with system dimension")

        # Initial guess
        phi_int = unxt.Quantity(
            jnp.zeros((coordinates.shape[0] - 2)), unit=phi_wall.unit
        )
        result = self.solver.solve(
            self.get_residual,
            phi_int,
            unxt.Quantity(jnp.full((1,), phi_wall.value), unit=phi_wall.unit),
            unxt.Quantity(jnp.zeros((1,)), unit=phi_wall.unit),
            coordinates,
        )
        result.set_coordinate(coordinates)

        result.set_solution(
            jnp.concatenate(
                [
                    unxt.Quantity(jnp.full((1,), phi_wall.value), unit=phi_wall.unit),
                    result.solution_int,
                    unxt.Quantity(jnp.zeros((1,)), unit=phi_wall.unit),
                ]
            )
        )

        self.result = result

    def get_residual(
        self,
        phi_int: unxt.Quantity,
        phi_left: unxt.Quantity,
        phi_right: unxt.Quantity,
        coordinates: unxt.Quantity,
    ):
        """
        Compute the Poisson-Boltzmann residual at interior grid nodes.

        Parameters
        ----------
        phi_int : unxt.Quantity, shape (N,)
            Electrostatic potential at the *N* interior grid nodes.
        phi_left : unxt.Quantity, shape (1,)
            Dirichlet value at the left boundary (electrode surface).
        phi_right : unxt.Quantity, shape (1,)
            Dirichlet value at the right boundary (bulk).
        coordinates : unxt.Quantity, shape (N+2,)
            Full 1-D grid coordinates including boundary nodes.

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

            \\rho_{\\mathrm{ion}} = N_A \\sum_j q_j c_j^0
                \\exp\\!\\left(-\\frac{q_j \\phi}{k_B T}\\right).

        Here :math:`q_j` is the ionic charge in Coulombs,
        :math:`c_j^0` is the bulk molar concentration,
        :math:`N_A` is Avogadro's number, and :math:`k_B` is the
        Boltzmann constant.
        The second derivative is approximated with a central
        finite-difference stencil.
        """
        # Assemble the full potential including boundary nodes and evaluate the
        # strong-form residual in SI units for consistent scaling.
        phi = jnp.concatenate(
            [
                phi_left,
                phi_int.reshape(-1),
                phi_right,
            ]
        )

        x = coordinates.to("m")
        phi = phi.to("V")
        dx_left = x[1:-1] - x[:-2]
        dx_right = x[2:] - x[1:-1]

        # Second derivative on a nonuniform 1-D grid.
        d2phi = (
            2.0
            / (dx_left + dx_right)
            * ((phi[2:] - phi[1:-1]) / dx_right - (phi[1:-1] - phi[:-2]) / dx_left)
        )

        # Generalized Boltzmann charge density term
        rho_ion = unxt.Quantity(0.0, "C / m^3")
        for ion in self.electrolyte.ions:
            charge = ion.charge.to("C")
            bulk_concentration = ion.molar_conc.to("mol/m^3")
            exponent = (
                -charge * phi[1:-1] / (_constants.BOLTZMANN_CONSTANT * self.temperature)
            ).to("")
            rho_ion += (
                charge
                * bulk_concentration
                * _constants.AVOGADRO_NUMBER
                * jnp.exp(exponent)
            )
        return d2phi + rho_ion / self.electrolyte.epsilon

    def get_ion_concentration_profiles(self):
        """
        Compute the local ion concentration profiles based on the computed
        potential.

        Returns
        -------
        conc_profiles : dict
            Mapping of ion name to local concentration profile as a
            :class:`unxt.Quantity` array with units of mol/m^3.

        Notes
        -----
        :meth:`compute` must be called before :meth:`get_ion_concentration_profiles`.
        """
        if self.result is None:
            raise ValueError(
                "Must call compute() before get_ion_concentration_profiles()"
            )

        conc_profiles = {}
        for ion in self.electrolyte.ions:
            valency = (ion.charge.to("C") / _constants.ELEMENTARY_CHARGE).to("").value
            conc_profiles[ion.name] = ion.molar_conc * boltzmann_factor(
                phi=self.result.solution, temperature=self.temperature, valency=valency
            )

        return conc_profiles

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
                * _constants.FARADAY_CONSTANT
                * self.solution
                / (_constants.GAS_CONSTANT * self.temperature)
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


"""
def get_residual(
        self,
        phi_int: unxt.Quantity,
        phi_left: unxt.Quantity,
        phi_right: unxt.Quantity,
        coordinates: unxt.Quantity,
    ):

        # Coordinates are provided in nm, while SI Poisson-Boltzmann is in meters.
        dx = (jnp.diff(coordinates)[1:] + jnp.diff(coordinates)[:-1]) / 2.0 * 1e-9
        # get full phi including boundaries
        phi = jnp.concatenate([jnp.array([phi_left]), phi_int, jnp.array([phi_right])])

        # Laplace term: d2phi/dx2
        d2phi = (phi[:-2] - 2 * phi[1:-1] + phi[2:]) / dx**2

        # Generalized Boltzmann charge density term
        # rho = sum( q_j * c_j * exp(-q_j * phi) )
        # todo: make the unit correct
        rho_ion = 0.0
        for ion in self.electrolyte.ions:
            z = ion.charge
            c_molm3 = ion.molar_conc.to("mol/m^3")
            rho_ion += (
                _constants.FARADAY_CONSTANT
                * z
                * c_molm3
                * jnp.exp(
                    -z
                    * _constants.FARADAY_CONSTANT
                    * phi[1:-1]
                    / (_constants.GAS_CONSTANT * self.temperature)
                )
            )

        # SI Poisson equation: d2phi/dx2 + rho/epsilon = 0
        return d2phi + rho_ion / self.electrolyte.epsilon

"""


def boltzmann_factor(
    phi: unxt.Quantity,
    temperature: unxt.Quantity,
    valency: int = 1,
):
    beta = 1.0 / (_constants.BOLTZMANN_CONSTANT * temperature).to("eV")
    return jnp.exp((-_constants.ELEMENTARY_CHARGE * phi * beta * valency).to(""))
