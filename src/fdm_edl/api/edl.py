# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import copy
import pathlib
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import unxt

if TYPE_CHECKING:
    from typing import Sequence

from ..models import ChargeModel, create_charge_model
from ..models.base import charge_density_profile
from ..models.solvent import BaseSolvent
from ..solver.base import BaseSolver, RootSolveResult
from ..solver.grad.base import BaseGradientOP
from ..utils import constants, load_dict
from ..utils import unit_conversion as uc
from ..utils.bc import BoundaryCondition
from ..utils.output_def import EDLStatus
from .electrode import Electrode
from .electrolyte import Electrolyte, Ion


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
    temperature : float
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

        self._unit_system = self._params.pop("unit", "metal")

        # get mandatory global parameters and set as attributes
        try:
            self._temperature: float = self._params[
                "temperature"
            ] * uc.get_conversion_factor("temperature", self._unit_system)
            self.temperature = unxt.Quantity(
                self._params["temperature"],
                unit=uc.UNIT_SYSTEMS[self._unit_system]["temperature"],
            )
        except KeyError:
            raise ValueError("`temperature` is required.")
        self.dim: int = self._params.pop("dim", 1)

        self.set_electrode(self._params.get("electrode", {}))
        self.set_electrolyte(self._params.get("electrolyte", {}))
        self.set_model(self._params.get("model", {}))
        self.set_solver(self._params.get("solver", {}))

        # placeholders for results
        self.result: EDLStatus | None = None
        self._solver_result: RootSolveResult | None = None

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
        self.electrode = Electrode(temperature=self.temperature)

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
            ``solvent`` : dict, optional
                Parameters for the solvent (default: ``{"name": "water"}``).
        """
        params = {
            "ions": {
                name: Ion(
                    name=name,
                    charge=unxt.Quantity(
                        data["charge"],
                        unit=uc.UNIT_SYSTEMS[self._unit_system]["electrical charge"],
                    ),
                    molar_conc=unxt.Quantity(
                        data["molar_conc"],
                        "mol / L",
                    ),
                    radius=(
                        unxt.Quantity(
                            data["radius"],
                            unit=uc.UNIT_SYSTEMS[self._unit_system]["length"],
                        )
                        if "radius" in data
                        else unxt.Quantity(
                            0.0, unit=uc.UNIT_SYSTEMS[self._unit_system]["length"]
                        )
                    ),
                )
                for name, data in _params.get("ions", {}).items()
            },
            "solvent": BaseSolvent(edl_obj=self, **_params.get("solvent", {})),
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
        self.solver = cast(BaseSolver, BaseSolver(**_params))

    def compute(
        self,
        coordinates: unxt.Quantity,
        boundary_conditions: Sequence[BoundaryCondition],
        grad_op: BaseGradientOP | None = None,
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
        grad_op : BaseGradientOP or None, optional
            Pre-built gradient operator.  When ``None`` (default), a
            :class:`~fdm_edl.utils.grad.BaseGradientOP` is
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
        # convert coordinates to internal unit system
        # Angstrom
        _coordinates = coordinates.to(uc.UNIT_SYSTEMS["metal"]["length"]).value
        # Volt
        _phi0 = (
            phi0.to(uc.UNIT_SYSTEMS["metal"]["electrical potential"]).value
            if phi0 is not None
            else None
        )

        # --- Normalise coordinates to (n_grid, n_dim) -----------------------
        if _coordinates.ndim == 1:
            coordinates_2d = _coordinates.reshape(-1, 1)
        else:
            coordinates_2d = _coordinates
        n_grid = coordinates_2d.shape[0]
        assert (
            coordinates_2d.shape[1] == self.dim
        ), "Coordinate dimension does not match model dimension."

        if self.dim != 1:
            raise NotImplementedError("Only dim=1 is currently supported.")

        coordinates_1d = coordinates_2d[:, 0]
        # --- Build gradient operator if not provided --------------------------------
        if grad_op is None:
            if self.dim == 1:
                # For 1-D, use the flat coordinate vector
                grad_op = BaseGradientOP()
            else:
                raise NotImplementedError(
                    f"Numerical gradient operator for dim={self.dim} is not yet implemented. "
                    "Pass a BaseGradientOP explicitly."
                )
        grad_fn = jax.jit(grad_op)

        # --- Zero initial guess if not provided ----------------
        if _phi0 is None:
            _phi0 = jnp.zeros((n_grid,))
        _phi0 = jnp.asarray(_phi0)

        # --- Solve -----------------------------------------------------------
        # for bc in boundary_conditions:
        #     # clamp Dirichlet nodes in the initial guess to improve convergence
        #     if bc._is_dirichlet:
        #         phi = bc.clamp_dirichlet(phi)
        self._solver_result = self.solver.solve(
            self.compute_residual,
            _phi0,
            boundary_conditions,
            coordinates_1d,
            grad_fn,
        )
        assert self._solver_result is not None

        #  --- post-process results ------------------------------------------------
        phi_final = self._solver_result.solution
        grad, _ = grad_fn(
            coordinates_1d, phi_final, h=coordinates_1d[1] - coordinates_1d[0]
        )
        efield = cast(
            unxt.Quantity,
            unxt.Quantity(
                -grad, unit=uc.UNIT_SYSTEMS["metal"]["electrical field strength"]
            ).to(uc.UNIT_SYSTEMS[self._unit_system]["electrical field strength"]),
        )
        ion_conc = self.charge_model.ion_concentration_profile(
            unxt.Quantity(
                phi_final,
                unit=uc.UNIT_SYSTEMS["metal"]["electrical potential"],
            ),
            self.electrolyte,
            self.temperature,
        )
        rho_ion = charge_density_profile(ion_conc)
        # calculate sigma from E-field at the electrode surface
        self.result = EDLStatus(
            coordinate=coordinates,
            sigma=cast(
                unxt.Quantity,
                (
                    self.electrolyte.solvent.eps_0
                    * constants.VACUUM_PERMITTIVITY
                    * efield[0]
                ).to(uc.UNIT_SYSTEMS[self._unit_system]["surface charge density"]),
            ),
            phi=cast(
                unxt.Quantity,
                unxt.Quantity(
                    phi_final,
                    unit=uc.UNIT_SYSTEMS["metal"]["electrical potential"],
                ).to(uc.UNIT_SYSTEMS[self._unit_system]["electrical potential"]),
            ),
            efield=efield,
            rho=cast(
                unxt.Quantity,
                rho_ion.to(
                    uc.UNIT_SYSTEMS[self._unit_system]["electrical charge density"]
                ),
            ),
            ion_conc=ion_conc,
        )

    def compute_residual(
        self,
        phi: jax.Array,
        boundary_conditions: Sequence[BoundaryCondition],
        coordinates: jax.Array,
        grad_op: BaseGradientOP,
    ):
        """
        Compute the Poisson-Boltzmann residual at all grid nodes.

        The physics equation ∇²φ + ρ_ion/ε = 0 is evaluated everywhere,
        then each boundary condition overwrites its nodes' residual entries
        with the appropriate constraint.
        Note: Use jax.Array rather than unxt.Quantity for the input and output of this function

        Parameters
        ----------
        phi : jax.Array, shape (n_grid,)
            Electrostatic potential at every grid node.
        grad_op : BaseGradientOP
            Discrete gradient operator built for this grid.
        boundary_conditions : sequence of BoundaryCondition
            Boundary conditions to enforce.
        coordinates : jax.Array, shape (n_grid,) or (n_grid, n_dim)
            Grid coordinates (2-D array).

        Returns
        -------
        residual : jax.Array, shape (n_grid,)
            Residual at each node.  Zero when the discretised
            Poisson-Boltzmann equation and all BCs are satisfied.
        """

        # --- Gradient of D-field -------------------------------------------------------
        grad, lap = grad_op(coordinates, phi, h=coordinates[1] - coordinates[0])
        grad_dfield = lap * self.electrolyte.solvent._eps_0

        # --- Ionic charge density (delegated to charge model) ----------------
        rho_ion = self.charge_model.charge_density(
            jnp.asarray(phi), self.electrolyte, self._temperature
        )

        # --- Physics residual at all nodes -----------------------------------
        # residual: e/Angstrom^3
        residual = -grad_dfield + rho_ion

        # --- Apply boundary conditions -----------
        for bc in boundary_conditions:
            residual = bc.update_residual(residual, phi, grad)

        return residual
