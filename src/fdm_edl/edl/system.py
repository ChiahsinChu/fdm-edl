# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import copy
import pathlib
from typing import Sequence

import quaxed.numpy as jnp
import unxt

from fdm_edl.bc import BoundaryCondition
from fdm_edl.edl.electrode import Electrode
from fdm_edl.edl.electrolyte import Electrolyte, Ion
from fdm_edl.edl.models import BikermanModel, ChargeModel, create_charge_model
from fdm_edl.operators import LaplacianOperator, LaplacianOperator1D
from fdm_edl.solver.base import BaseSolver
from fdm_edl.solver.optax_solver import OptaxSolver
from fdm_edl.utils import load_dict, to_unxtq

from .. import _constants


class ElectricalDoubleLayer:
    """
    Electrical Double Layer (EDL) model based on the Poisson-Boltzmann
    equation, solved with a finite-difference method.

    Supports 1-D, 2-D, and 3-D grids.  Coordinates are expected as
    ``(n_grid, n_dim)``; 1-D arrays ``(n_grid,)`` are accepted and
    reshaped automatically.

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
    result : RootSolveResult or None
        Full solver result from the last :meth:`compute` call, or ``None``
        before :meth:`compute` is called.  The potential profile is
        ``result.solution`` and the grid coordinates ``result.coordinate``.

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
        self.set_model(self._params.get("model", {}))
        self.set_solver(self._params.get("solver", {}))

        # placeholders for results
        self.result = None
        self._apply_bcs = True

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
                    **(
                        {"radius": to_unxtq(data["radius"])} if "radius" in data else {}
                    ),
                )
                for name, data in _params.get("ions", {}).items()
            ],
            "epsilon": _params.get("epsilon_r", 78.5) * _constants.VACUUM_PERMITTIVITY,
            "temperature": self.temperature,
        }
        self.electrolyte = Electrolyte(**params)

    def set_model(self, _params: dict):
        """Set the charge-density model.

        Parameters
        ----------
        _params : dict
            Must contain ``"type"`` (str) matching a registered model
            name (default ``"boltzmann"``).  Remaining keys are forwarded
            to the model constructor.
        """
        self.charge_model: ChargeModel = create_charge_model(_params)

    def set_solver(self, _params: dict):
        """
        Set up the nonlinear solver with new parameters.

        Parameters
        ----------
        _params : dict
            Keyword arguments forwarded to :class:`~fdm_edl.solver.BaseSolver`.
        """
        self.solver = BaseSolver(**_params)

    def compute(
        self,
        coordinates: unxt.Quantity,
        boundary_conditions: Sequence[BoundaryCondition],
        laplacian: LaplacianOperator | None = None,
        phi0: unxt.Quantity | None = None,
    ):
        """
        Solve for the electrostatic potential profile.

        Parameters
        ----------
        coordinates : unxt.Quantity, shape (n_grid,) or (n_grid, n_dim)
            Grid coordinates.  1-D arrays are accepted and reshaped to
            ``(n_grid, 1)`` internally.
        boundary_conditions : list of BoundaryCondition
            A list of :class:`~fdm_edl.bc.BoundaryCondition` objects that
            define the constraints on specific nodes.
        laplacian : LaplacianOperator or None, optional
            Pre-built Laplacian operator.  When ``None`` (default), a
            :class:`~fdm_edl.operators.LaplacianOperator1D` is
            constructed automatically for 1-D problems.
        phi0 : unxt.Quantity or None, optional
            Initial guess for the potential at all grid nodes, shape
            ``(n_grid,)``.  When ``None`` (default), a zero array is
            created and seeded with boundary-condition values.

        Returns
        -------
        None
            The result is stored in ``self.result``.
        """
        # --- Normalise coordinates to (n_grid, n_dim) -----------------------
        if coordinates.ndim == 1:
            coordinates_2d = coordinates.reshape(-1, 1)
        else:
            coordinates_2d = coordinates
        n_grid = coordinates_2d.shape[0]

        # --- Build Laplacian if not provided --------------------------------
        if laplacian is None:
            if self.dim == 1:
                # For 1-D, use the flat coordinate vector
                laplacian = LaplacianOperator1D(coordinates_2d[:, 0])
            else:
                raise NotImplementedError(
                    f"Automatic Laplacian for dim={self.dim} is not yet implemented. "
                    "Pass a LaplacianOperator explicitly."
                )

        # --- Initial guess (zeros, then seed with BC values) ----------------
        if phi0 is None:
            phi0 = unxt.Quantity(jnp.zeros((n_grid,)), unit="V")
        for bc in boundary_conditions:
            phi0 = bc.apply_initial_guess(phi0)

        # --- Solve -----------------------------------------------------------
        use_penalty = (
            isinstance(self.solver, OptaxSolver)
            and self.solver.bc_enforcement == "penalty"
        )
        if use_penalty:
            self._apply_bcs = False
        try:
            result = self.solver.solve(
                self.get_residual,
                phi0,
                laplacian,
                boundary_conditions,
                coordinates_2d,
                **(
                    {
                        "boundary_conditions": boundary_conditions,
                        "coordinates": coordinates_2d,
                    }
                    if use_penalty
                    else {}
                ),
            )
        finally:
            self._apply_bcs = True
        result.set_coordinate(coordinates)
        result.set_solution(result.solution_int)

        self.result = result

    def get_residual(
        self,
        phi: unxt.Quantity,
        laplacian: LaplacianOperator,
        boundary_conditions: Sequence[BoundaryCondition],
        coordinates: unxt.Quantity,
    ):
        """
        Compute the Poisson-Boltzmann residual at all grid nodes.

        The physics equation ∇²φ + ρ_ion/ε = 0 is evaluated everywhere,
        then each boundary condition overwrites its nodes' residual entries
        with the appropriate constraint.

        Parameters
        ----------
        phi : unxt.Quantity, shape (n_grid,)
            Electrostatic potential at every grid node.
        laplacian : LaplacianOperator
            Discrete Laplacian operator built for this grid.
        boundary_conditions : sequence of BoundaryCondition
            Boundary conditions to enforce.
        coordinates : unxt.Quantity, shape (n_grid, n_dim)
            Grid coordinates (2-D array).

        Returns
        -------
        residual : unxt.Quantity, shape (n_grid,)
            Residual at each node.  Zero when the discretised
            Poisson-Boltzmann equation and all BCs are satisfied.
        """
        # --- Laplacian -------------------------------------------------------
        d2phi = laplacian(phi)

        # --- Ionic charge density (delegated to charge model) ----------------
        rho_ion = self.charge_model.charge_density(
            phi, self.electrolyte, self.temperature
        )

        # --- Physics residual at all nodes -----------------------------------
        residual = d2phi + rho_ion / self.electrolyte.epsilon

        # --- Apply boundary conditions -----------
        if self._apply_bcs:
            # Flatten coordinates for 1-D BCs that expect a 1-D array
            coords_for_bc = (
                coordinates[:, 0]
                if coordinates.ndim == 2 and coordinates.shape[1] == 1
                else coordinates
            )
            for bc in boundary_conditions:
                residual = bc.apply_residual(residual, phi, coords_for_bc)

        return residual

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

        phi = self.result.solution

        # Bikerman needs the full electrolyte to compute the denominator
        if isinstance(self.charge_model, BikermanModel):
            return self.charge_model.ion_concentration_full(
                phi, self.electrolyte, self.temperature
            )

        conc_profiles = {}
        for ion in self.electrolyte.ions:
            conc_profiles[ion.name] = self.charge_model.ion_concentration(
                phi, ion, self.temperature
            )
        return conc_profiles

    # def plot(self):
    #     """
    #     Plot the electrostatic potential and ion concentration profiles.

    #     Returns
    #     -------
    #     fig : matplotlib.figure.Figure
    #         The figure object containing both subplots.
    #     axs : ndarray of matplotlib.axes.Axes, shape (2,)
    #         Array of axes: ``axs[0]`` holds the potential profile and
    #         ``axs[1]`` the ion concentration profiles.

    #     Notes
    #     -----
    #     :meth:`compute` must be called before :meth:`plot`.
    #     """
    #     nrows = 2
    #     ncols = 1
    #     fig, axs = plt.subplots(
    #         nrows, ncols, figsize=(nrows * 4, ncols * 5), sharex=True
    #     )

    #     # Plot Potential
    #     ax = axs[0]
    #     ax.plot(self.coordinates, self.solution, color="black", lw=2)
    #     ax.set_title(r"Electrostatic Potential $\phi(x)$")
    #     ax.set_ylabel(r"$\phi$ (V)")
    #     ax.grid(True)

    #     # Plot Concentrations
    #     ax = axs[1]
    #     total_q_conc = 0.0
    #     for name, data in self.electrolyte.ions.items():
    #         z = data["charge"]
    #         c0 = data["conc"]
    #         # Local concentration in mol/L.
    #         local_c = c0 * jnp.exp(
    #             -z
    #             * _constants.FARADAY_CONSTANT
    #             * self.solution
    #             / (_constants.GAS_CONSTANT * self.temperature)
    #         )
    #         ax.plot(self.coordinates, local_c, label=f"{name} ($z={z}$)")
    #         total_q_conc += local_c * z

    #     ax.plot(
    #         self.coordinates,
    #         total_q_conc,
    #         label="Total Charge Concentration",
    #         color="red",
    #         lw=2,
    #         linestyle="--",
    #     )

    #     ax.set_title("Ion Concentrations")
    #     ax.set_ylabel("Concentration (M)")
    #     ax.legend()
    #     ax.grid(True)

    #     ax.set_xlabel("nm")
    #     plt.tight_layout()
    #     return fig, axs


def boltzmann_factor(
    phi: unxt.Quantity,
    temperature: unxt.Quantity,
    valency: int = 1,
):
    """Compute the Boltzmann factor exp(-z e φ / k_B T).

    Parameters
    ----------
    phi : unxt.Quantity
        Electrostatic potential.
    temperature : unxt.Quantity
        Absolute temperature.
    valency : int, optional
        Ion valency *z* (default: 1).

    Returns
    -------
    unxt.Quantity
        Dimensionless Boltzmann factor at each grid node.
    """
    beta = 1.0 / (_constants.BOLTZMANN_CONSTANT * temperature).to("eV")
    return jnp.exp((-_constants.ELEMENTARY_CHARGE * phi * beta * valency).to(""))
