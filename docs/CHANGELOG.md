---
status: draft
author: Jia-Xin Zhu, AI Agent
last_updated: 2026-05-17
---

# CHANGELOG

## [Unreleased] - 2026-05-17

### Added

- GitHub Pages deployment workflow in `.github/workflows/deploy-docs-ghpages.yml` to build VitePress docs and publish `docs/.vitepress/dist` on pushes to `master`
- New solver diagnostics in `RootSolveResult` (`gradient`, `source`) in `src/fdm_edl/solver/base.py`
- New `src/fdm_edl/solver/newton.py` implementation with configurable backtracking line search (`alpha`, `max_iter_ls`) and multi-criterion convergence controls (`atol_*`, `rtol_*`)
- New matrix-free Krylov linear solver module `src/fdm_edl/solver/scipy.py` with `BiCGStabSolver`, `CGSolver`, and `GMRESSolver` implementations using `jax.scipy.sparse.linalg`
- New solver-regression coverage in `tests/test_solver.py` to validate Newton/BiCGStab/CG/GMRES consistency against the nonlinear Poisson-Boltzmann benchmark
- VitePress docs scaffold via `docs/.vitepress/config.mts` and npm scripts in `package.json` (`docs:dev`, `docs:build`, `docs:preview`)
- `docs/generate_api_docs.py` to generate pdoc HTML API pages into `docs/public/api/`
- Static API pages under `docs/public/api/` and VitePress reference entry pages (`docs/reference/index.md`, `docs/reference/fdm_edl.md`)
- New Bikerman example assets in `examples/01.bikerman/` (`input_booth.json`, `input_uniform.json`, `run.py`, `run.ipynb`, `booth.txt`, `uniform.txt`)

### Changed

- Python test CI workflow in `.github/workflows/test_python.yml` now runs on push/pull_request with Python 3.11-3.13, adds disk-space cleanup, emits JUnit output, and uploads both coverage and test-results reports through Codecov v5
- VitePress config in `docs/.vitepress/config.mts` now uses title `FDM-EDL`, excludes internal docs-agent files from source pages, and ignores generated API/source dead-link patterns
- API reference redirect page `docs/reference/fdm_edl.md` switched from inline script to Vue `onMounted()` redirect logic for VitePress compatibility
- Docs optional dependency set in `pyproject.toml` was simplified to `pdoc` only, reflecting the current API docs generation pipeline
- Solver base API now accepts tolerance families (`atol_var`, `rtol_var`, `atol_grad`, `rtol_grad`, `atol_src`, `rtol_src`, `atol_res`, `rtol_res`) and provides shared helper routines for residual norms, Dirichlet clamping, and loop conditions
- Extracted convergence-check logic into shared `_convergence_flag()` helper method in `BaseSolver`, eliminating duplicate conditional logic across `NewtonSolver` and `ScipySolver` implementations
- `ElectricalDoubleLayer._loss()` now returns residual plus auxiliary fields (`grad(phi)`, `div_D`, `rho_ion`) so solvers can use richer convergence diagnostics
- `ElectricalDoubleLayer` now converts solver absolute tolerances from the configured unit system into internal metal units before solver construction
- Legacy Newton module `src/fdm_edl/solver/naive_newton.py` replaced by `src/fdm_edl/solver/newton.py`, and solver imports updated accordingly
- Solver exports in `src/fdm_edl/solver/__init__.py` now import Krylov implementations from `scipy.py` (`BiCGStabSolver`, `CGSolver`, `GMRESSolver`) instead of the temporary module path
- Solver fixtures in `tests/data/*.json` migrated from `tol` to the new tolerance keys (`atol_var`, with `method`/`atol_res` where needed)
- `serialize()` on `EDLStatus` and `IsothermStatus` now returns structured dataclass fields directly; serialization helper utilities were extended in `src/fdm_edl/utils/output_def.py`
- README was rewritten to reflect the current public API, solver/model options, and parameter schema
- `docs/index.md` now includes README content directly for single-source documentation

### Fixed

- Type signatures now consistently describe residual functions with auxiliary outputs across API and solver modules, improving static typing for Newton/JAX Jacobian calls

### Removed

- Legacy MkDocs configuration `mkdocs.yml`
- Legacy reference generator `docs/gen_reference.py`

## [Unreleased] - 2026-04-27

### Added

- `save_dict` in `src/fdm_edl/utils/io.py` for saving a parameter dict to JSON or YAML; exported from `src/fdm_edl/utils/__init__.py`
- `serialize()` and `deserialize()` methods on `EDLStatus` and `IsothermStatus` in `src/fdm_edl/utils/output_def.py` for round-trip JSON/YAML persistence
- `epsilon_r` on `EDLStatus`, including serialization support, so solved dielectric profiles can be persisted with the electrostatic solution

### Changed

- `ElectricalDoubleLayer` now recomputes electrode boundary-condition coefficients from the local field-dependent dielectric response before each residual evaluation
- Surface charge output in `ElectricalDoubleLayer` is now derived from the interfacial dielectric response `eps(E_OHP)` instead of the bulk-water permittivity constant
- Solvent permittivity evaluation now uses `|E|` consistently so Booth and Langevin dielectric models remain even in field direction
- Booth and Langevin dielectric closures now share a stable `L(x) / x` evaluation path to improve behavior across the small-field transition

### Fixed

- Numerical stability in `BaseIsotherm.residual_fn`: normalize coverage to `y = theta / theta_max`, clamp to `(eps, 1-eps)`, and evaluate `log(y) - log1p(-y)` to avoid overflow near the coverage boundaries
- Removed stray debug `print` calls in `BaseIsotherm` optimization loop
- Removed small-field dielectric overshoot in Booth and Langevin solvent models by replacing cancellation-prone formulas with polynomial limits near zero field
- Added regression coverage in `tests/test_water_eps.py` for dielectric-response transition regions where the overshoot had been observed

## [Unreleased] - 2026-04-23

### Added

- New `EuclideanFVOp` in `src/fdm_edl/solver/grad/conservative_flux.py` for conservative finite-volume evaluation of `div(eps * E)` on 1D grids
- Exported `EuclideanFVOp` from `src/fdm_edl/solver/grad/__init__.py`
- New operator package `fdm_edl.op` with unified exports and a `create_gradient_op()` factory for discretization/coordinate-system dispatch
- New axisymmetric operator classes `AxisymmetricFDOp` and `AxisymmetricFVOp` in `src/fdm_edl/op/axisymmetric/`
- New cartesian operator modules in `src/fdm_edl/op/cartesian/` with shared base API in `src/fdm_edl/op/base.py`

### Changed

- `ElectricalDoubleLayer` now selects gradient operators by solvent type in 1D: `EuclideanFDOp` for `uniform`, `EuclideanFVOp` for `langevin`/`booth`
- PB residual in `ElectricalDoubleLayer._loss()` is now expressed consistently as `div_D - rho_ion`, with boundary-condition updates using the returned `grad(phi)` field
- `BaseGradientOP` now carries an `eps_func` callback and returns `(grad_phi, div_D)` from its public call interface
- `EuclideanFDOp` now computes `div_D` via dielectric-aware scaling of the Laplacian using `eps_func`
- Solvent subclasses now declare canonical registry keys via class attribute `type` (for example: `uniform`, `langevin`, `booth`)
- `BoothDielectrics` type registration corrected to `booth` and its `ElectricalDoubleLayer` import moved under `TYPE_CHECKING`
- Finite-difference and finite-volume divergence operators now include axisymmetric cylindrical radial forms (`d2/dr2 + (1/r)d/dr` and `(1/r)d(rD_r)/dr`) with regular handling near `r=0`
- Legacy gradient modules now support coordinate-system-aware divergence via `coordinate_system` selection

### Fixed

- Updated `tests/test_grad.py` to validate `div_D` outputs against autodiff Laplacian with dielectric scaling, matching the new gradient-operator API

## [0.1.dev11] - 2026-04-21

### Added

- New `fdm_edl.solver.grad` package with gradient/Laplacian operator infrastructure
- [`BaseGradientOP`](src/fdm_edl/solver/grad/base.py:15) class for numerical gradient operators with JIT-friendly design
- [`EuclideanFDOp`](src/fdm_edl/solver/grad/laplacian.py:15) class for numerical Laplacian (second derivative) operators
- Support for both uniform and nonuniform grids with configurable boundary/interior stencil points
- Export of gradient operators in [`fdm_edl.solver.grad`](src/fdm_edl/solver/grad/__init__.py:1)

### Changed

- [`ElectricalDoubleLayer`](src/fdm_edl/api/edl.py:25) now uses [`EuclideanFDOp`](src/fdm_edl/solver/grad/laplacian.py:15) as the default gradient operator for 1D cases
- Removed explicit `h` parameter from gradient operator calls in [`ElectricalDoubleLayer.compute()`](src/fdm_edl/api/edl.py:289)
- Fixed sign in D-field gradient calculation: `grad_dfield = -lap * self.electrolyte.solvent._eps_0`

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

- Updated `ElectricalDoubleLayer` to use solvent permittivity in PB residual formulation
- Improved type annotations in solver and API modules

## [0.1.dev7] - 2026-04-18

### Added

- `LangevinWater` class for field-dependent water permittivity based on Langevin theory
- `WaterEpsTester` mixin for validating water permittivity models
- `langevin_eps()` function for Langevin-based dielectric response

### Changed

- Refactored solvent permittivity calculations to support field-dependent models
- Updated PB solver to use solvent permittivity in the D-field formulation

## [0.1.dev6] - 2026-04-17

### Added

- `Electrolyte` class with support for multi-ion electrolytes
- `ChargeModel` base class with registry-based implementation

### Changed

- Refactored PB solver to support general charge models
- Improved unit handling throughout the codebase

## [0.1.dev5] - 2026-04-16

### Added

- Initial public release
- Basic 1D Poisson-Boltzmann solver
- Analytical comparison with Gouy-Chapman theory

## Changelog

| Version    | Date       | Changes                                                | Author      |
| ---------- | ---------- | ------------------------------------------------------ | ----------- |
| v0.1.dev11 | 2026-04-21 | Added gradient/Laplacian operator infrastructure       | AI Agent    |
| v0.1.dev10 | 2026-04-21 | Added solvent-model package with Langevin/Booth models | Jia-Xin Zhu |
| v0.1.dev9  | 2026-04-19 | Added Booth dielectric model and test infrastructure   | Jia-Xin Zhu |
| v0.1.dev8  | 2026-04-19 | Added Solvent dataclass and explicit solvent blocks    | Jia-Xin Zhu |
| v0.1.dev7  | 2026-04-18 | Added Langevin water permittivity model                | Jia-Xin Zhu |
| v0.1.dev6  | 2026-04-17 | Added multi-ion electrolyte support                    | Jia-Xin Zhu |
| v0.1.dev5  | 2026-04-16 | Initial public release                                 | Jia-Xin Zhu |
