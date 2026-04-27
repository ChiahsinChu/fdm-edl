---
status: draft
author: Jia-Xin Zhu, AI Agent
last_updated: 2026-04-27
---

# Development Plans

## Features

### General

- [x] `ElectricalDoubleLayer` base class with `compute()` and `plot()` methods (task.000)
- [x] SI-unit multi-ion Poisson-Boltzmann solver
- [x] Solvent model framework (`BaseSolvent`) with pluggable dielectric response types
- [x] Load parameters from `dict`, JSON, or YAML file
- [x] Save parameters / output results to JSON or YAML via `save_dict` (`utils/io.py`)
- [x] `serialize()`/`deserialize()` on `EDLStatus` and `IsothermStatus` for round-trip JSON/YAML persistence
- [x] Electroneutrality check on electrolyte input
- [x] Compute analytical properties (e.g., Debye length) without solving the full PDE
- [x] Add physical units to output results and plots (e.g., via [`unxt`](https://github.com/GalacticDynamics/unxt))
- [x] Improve type annotations and static-type compatibility in API/model/solver modules

### Charge models

- [ ] Bikerman model with mixed cation/anion types
- [x] Refactor solvent dielectric models into `models.solvent` package with registry dispatch
- [x] Add solvent response (Booth)
- [x] Add solvent response (Langevin)
- [x] Bikerman model with single cation/anion types
- [x] Stabilize Booth/Langevin dielectric response near zero field and enforce even response with respect to field sign

### Boundary conditions

- [ ] multi chemisorption equilibrium
- [x] single chemisorption (Frumkin isotherm, known: lateral coefficient & potential)
- [x] Add safer BC residual application and public Dirichlet accessor (`is_dirichlet`)
- [x] Update field-dependent boundary-condition coefficients from the local solvent permittivity during residual evaluation

### Optimizer

- [ ] Add general interfaces to optax

### 2D/3D FEM

- [ ] Add interface to [JAX-FEM](https://github.com/deepmodeling/jax-fem) for 2D/3D cases

### Nanotubes

- [x] OPs for cylindrical coordinates

### Gradient Operators

- [x] Add gradient/Laplacian operator infrastructure in `fdm_edl.solver.grad`
- [x] Implement [`BaseGradientOP`](src/fdm_edl/solver/grad/base.py:15) for numerical gradient operators with JIT-friendly design
- [x] Implement [`EuclideanFDOp`](src/fdm_edl/solver/grad/laplacian.py:15) for numerical Laplacian operators
- [x] Implement [`EuclideanFVOp`](src/fdm_edl/solver/grad/conservative_flux.py:1) for conservative finite-volume `div(eps * E)` evaluation
- [x] Support for both uniform and nonuniform grids with configurable stencil points
- [x] Default to [`EuclideanFDOp`](src/fdm_edl/solver/grad/laplacian.py:15) in [`ElectricalDoubleLayer`](src/fdm_edl/api/edl.py:25)
- [x] Route default 1D gradient operator by solvent type (`uniform` -> `EuclideanFDOp`; `langevin`/`booth` -> `EuclideanFVOp`)
- [x] Add coordinate-system-aware operator package `fdm_edl.op` with cartesian/axisymmetric FD+FV implementations and factory dispatch
- [ ] Validate and document the mapping between gradient operators and solvent dielectric models
- [ ] Add focused regression tests for axisymmetric `r=0` behavior in FD/FV divergence operators

## Documentation

- [x] Add usage example notebook for 1D EDL (extend `examples/00.1d_edl/run.ipynb`)
- [x] Add API docstrings to `ElectricalDoubleLayer`, `Electrode`, `Electrolyte`, `Solver`
- [x] Add API docstrings to isotherm module (`BaseIsotherm`, `LangmuirIsotherm`, `FrumkinIsotherm`)

## Unit Tests

- [ ] Add coverage-driven tests for PBE residual pathways (focus: `benchmark/pb1d.py`, uncovered branches at 60% coverage)
- [x] Add numerical-consistency tests for gradient/Laplacian operators against JAX autodiff (`tests/test_grad.py`)
- [x] Add dielectric-response regression tests around the small-field transition for Booth and Langevin water models (`tests/test_water_eps.py`)
- [ ] Add dedicated unit tests for `EuclideanFVOp` on nonuniform grids and nonlinear dielectric closures
- [ ] Add operator regression tests for low-coverage coordinate operators: `op/cartesian/fd.py`, `op/cartesian/fv.py`, `op/axisymmetric/fd.py`, `op/axisymmetric/fv.py`
- [ ] Add first-pass tests for currently untested modules: `_version.py`, `benchmark/base.py`, `benchmark/gcs1d.py`, `isotherm/base.py`, `isotherm/frumkin.py`, `isotherm/langmuir.py`, `utils/grad.py`
- [ ] Add branch/validation tests for API and utility gaps: `api/bc.py`, `api/edl.py`, `utils/bc.py`, `utils/io.py`
- [ ] Add model-behavior tests for `models/bikerman.py` and `models/solvent/base.py` edge cases

## Chores

- [x] Set up CI from GitHub workflow files (https://docs.gitlab.com/ci/migration/github_actions/)
- [ ] Check GPU compatibility for JAX solver
