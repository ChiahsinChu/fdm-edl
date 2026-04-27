---
status: draft
author: Jia-Xin Zhu, AI Agent
last_updated: 2026-04-27
---

# CHANGELOG

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
