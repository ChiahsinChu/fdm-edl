---
status: draft
author: Jia-Xin Zhu, AI Agent
last_updated: 2026-04-19
---

# CHANGELOG

## [0.1.dev7] - 2026-04-19

### Added

- GitLab CI pipeline configuration in `.gitlab-ci.yml` with multi-version Python test jobs (`3.11`, `3.12`, `3.13`)
- JUnit test report artifact export (`report.xml`) for unit-test jobs
- Coverage stage with `pytest --cov=fdm_edl tests/` and GitLab coverage regex parsing
- Static type-check stage running `mypy src/fdm_edl`

### Changed

- Standardized CI default image and setup steps to install dependencies via `pip install ".[all]"` before each job

## [0.1.dev6] - 2026-04-19

### Added

- Type-stub dependency `types-PyYAML` in project dependencies to improve static checking support
- Public `BoundaryCondition.is_dirichlet` property for safer boundary-condition handling in solvers

### Changed

- Tightened type annotations across core modules (`api`, `models`, `isotherm`, `benchmark`, `utils`) with explicit optional/result typing, `cast(...)` usage, and improved return signatures
- Refined `ElectricalDoubleLayer` solver pipeline to use explicitly JIT-compiled gradient function naming and array-normalized initial guesses
- Simplified package top-level metadata in `fdm_edl.__init__` by removing eager submodule exports and clarifying package scope in the docstring
- Updated benchmark helpers for improved scalar/array handling and typed external imports

### Fixed

- `BoundaryCondition.apply` now handles missing gradients for Dirichlet conditions and raises clear errors for non-Dirichlet usage without gradient input
- Newton solver now relies on the public Dirichlet flag instead of private internals when clamping boundary values
- `charge_density_profile` now validates empty concentration inputs explicitly and avoids fragile array summation behavior

## [0.1.dev5] - 2026-04-17

### Added

- Ion molar volume computation in `Ion` dataclass for steric effect calculations
- Comprehensive NumPy-style docstrings to boundary condition classes and Bikerman model
- Improved type annotations with `TYPE_CHECKING` for forward references across API modules

### Changed

- **BC API refactoring**: Converted from Protocol-based factory functions to ABC-based factory class pattern
  - Introduced `TemplateBC` base class for standardized boundary condition coefficient providers
  - Converted `make_constp_bc()` → `ConstP` class, `make_neumann_bc()` → `NeumannBC` class, etc.
  - All BC classes now store coefficients in `self.coeff` and implement `make_coeff()` method
- **Bikerman model improvements**: Refactored steric effects calculation for better numerical stability
  - Replaced list-based density calculations with JAX array operations
  - Improved handling of packing fraction denominator computation
  - Renamed `ion_concentration_full()` to `ion_concentration_profile()` for clarity
  - Enhanced docstrings with parameter and return value documentation
- Import reorganization: Prioritized `from jax import numpy` over `quaxed.numpy` for consistency
- Updated solver modules with improved type hints and docstrings

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
