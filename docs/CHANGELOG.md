---
status: draft
author: Jia-Xin Zhu, AI Agent
last_updated: 2026-04-16
---

# CHANGELOG

## [0.1.dev4] - 2026-04-16

### Added

- NumPy-style docstrings for the isotherm module: `BaseIsotherm`, `LangmuirIsotherm`, `FrumkinIsotherm`, and all public methods

## [0.1.dev3] - 2026-03-25

### Added

- Boundary condition module (`bc/`): `BoundaryCondition` ABC, `DirichletBC`, `NeumannBC`, `RobinBC`, `PeriodicBC`
- Laplacian operator module (`operators/`): `LaplacianOperator` ABC, `LaplacianOperator1D` (3-point nonuniform stencil)
- Optional `phi0` parameter in `compute()` for user-supplied initial guesses

### Changed

- Generalised `ElectricalDoubleLayer.compute()` to accept `(n_grid, n_dim)` coordinates (1-D arrays auto-reshaped)
- `compute()` now takes a `list[BoundaryCondition]` instead of a scalar `phi_wall`; Laplacian is an optional external operator
- `get_residual()` evaluates physics at all nodes and lets BCs overwrite their entries
- Solver operates on the full grid vector (boundary + interior) instead of interior-only
- Updated test to use explicit `DirichletBC` objects

## [0.1.dev2] - 2026-03-17

### Added

- `ElectricalDoubleLayer` class (`edl/system.py`) as the main user-facing API for 1D EDL simulations
- SI-unit Poisson-Boltzmann solver supporting multi-ion electrolytes
- `epsilon_r` parameter (default 78.5 for water) for solvent dielectric constant
- `Electrode`, `Electrolyte`, and `Solver` sub-classes
- Parameters can be loaded from a `dict`, JSON file, or YAML file
- `compute(coordinates, phi_wall)` method to solve the electrostatic potential profile
- `plot()` method to visualize the potential and local ion concentration profiles
- Electroneutrality check on the input electrolyte composition
- Physical constants module (`utils/constants.py`)

### Changed

- Refactored solver to return a structured result object; solution is persisted on the `ElectricalDoubleLayer` instance as `edl.solution`

## [0.1.dev1] - 2026-03-11

### Added

- Initial repository scaffold: `pyproject.toml`, package skeleton, `.gitignore`
- Prototype dimensionless 1D PBE solver (`workspace/task.000/run.py`) using JAX and Newton iteration
