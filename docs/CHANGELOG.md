---
status: draft
author: Jia-Xin Zhu, AI Agent
last_updated: 2026-04-21
---

# CHANGELOG

## [0.1.dev10] - 2026-04-21

### Added

- New solvent-model package `fdm_edl.models.solvent` with a registry-based `BaseSolvent` interface and default `UniformDielectrics` implementation
- Field-dependent solvent classes `LangevinDielectrics` and `BoothDielectrics` (with `LangevinWater` and `BoothWater` presets)
- `tests/data/NaCl_langevin.json` input fixture for Langevin-type solvent configuration

### Changed

- `ElectricalDoubleLayer` now builds electrolyte solvent objects via `BaseSolvent(...)` and accepts solvent configs keyed by `type` (for example, `uniform` or `langevin`)
- `Electrolyte.solvent` type changed from the legacy API `Solvent` dataclass to `BaseSolvent`
- PB residual now uses the D-field form (`-grad(eps*grad(phi)) + rho`) using the solvent permittivity scale
- Gradient-operator typing in `ElectricalDoubleLayer` generalized to `BaseGradientOP`
- Langevin/Booth utilities and tests migrated from `utils` into `models.solvent` APIs
- Test data schema updated from `eps_0`/`eps_opt` to model-specific keys such as `type` and `epsilon_r`

### Removed

- Legacy `fdm_edl.utils.langevin` module and legacy inline `Solvent` dataclass in `fdm_edl.api.electrolyte`

## [0.1.dev9] - 2026-04-19

### Added

- `booth_eps()` function for field-dependent dielectric response using Booth's saturation model
- `BoothWaterEps` class for field-dependent water permittivity based on Booth theory
- Reusable `WaterEpsTester` test mixin and dedicated Booth-model test coverage in `tests/test_langevin_water.py`

### Changed

- Refactored `langevin_eps()` with shared precomputed coefficients for clearer and more efficient evaluation
- Reorganized Langevin dielectric tests to share validation logic across multiple water permittivity models

## [0.1.dev8] - 2026-04-19

### Added

- `Solvent` dataclass in `fdm_edl.api.electrolyte` with static (`eps_0`) and optical (`eps_opt`) relative permittivities plus internal unit-system conversions
- NumPy-style API docstring for `Solvent`
- Explicit `solvent` blocks in example and test electrolyte JSON inputs

### Changed

- `Electrolyte` now stores solvent properties via `solvent: Solvent` instead of scalar `epsilon`/`epsilon_r` fields
- `ElectricalDoubleLayer` electrolyte parameter parsing now constructs a `Solvent` object and accepts a `solvent` dictionary in input files
- Electrostatics/benchmark paths now use `electrolyte.solvent.eps_0` for permittivity-dependent calculations (residual, Debye length, sigma, PB analytical profiles)
- GCS example notebook and Z-Z GCS test updated to use the new solvent-based API

## [0.1.dev7] - 2026-04-19

### Added

- GitLab CI pipeline configuration in `.gitlab-ci.yml` with multi-version Python test jobs (`3.11`, `3.12`, `3.13`)
- JUnit test report artifact export (`report.xml`) for unit-test jobs
- Coverage stage with `pytest --cov=fdm_edl tests/` and GitLab coverage regex parsing
- Static type-check stage running `mypy src/fdm_edl`
- `LangevinWaterEps` class for field-dependent dielectric response modeling of liquid water
- `langevin_eps()` function to compute field-dependent dielectric contributions from dipolar alignment
- Unit tests for Langevin water model in `tests/test_langevin_water.py`
- Unit tests for nonlinear Poisson-Boltzmann solver in `tests/test_nonlinear_pb.py`
- Unit tests for Z-Z GCS model in `tests/_test_z-z_gcs.py`

### Changed

- Standardized CI default image and setup steps to install dependencies via `pip install ".[all]"` before each job
- Temporarily disabled type-check CI stage pending further configuration
- Reorganized development todo list with completed tasks moved to their respective sections

### Fixed

- Fixed ambiguous variable name `l` → `langevin_val` in `langevin_eps()` (E741 lint error)

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
