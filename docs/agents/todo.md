---
status: draft
author: Jia-Xin Zhu, AI Agent
last_updated: 2026-04-21
---

# Development Plans

## Features

### General

- [x] `ElectricalDoubleLayer` base class with `compute()` and `plot()` methods (task.000)
- [x] SI-unit multi-ion Poisson-Boltzmann solver
- [x] Solvent model framework (`BaseSolvent`) with pluggable dielectric response types
- [x] Load parameters from `dict`, JSON, or YAML file
- [x] Electroneutrality check on electrolyte input
- [x] Compute analytical properties (e.g., Debye length) without solving the full PDE
- [x] Add physical units to output results and plots (e.g., via [`unxt`](https://github.com/GalacticDynamics/unxt))
- [x] Improve type annotations and static-type compatibility in API/model/solver modules

### Charge models

- [x] Add solvent response (Booth)
- [x] Refactor solvent dielectric models into `models.solvent` package with registry dispatch
- [ ] Bikerman model with mixed cation/anion types
- [x] Add solvent response (Langevin)
- [x] Bikerman model with single cation/anion types

### Boundary conditions

- [ ] multi chemisorption equilibrium
- [x] single chemisorption (Frumkin isotherm, known: lateral coefficient & potential)
- [x] Add safer BC residual application and public Dirichlet accessor (`is_dirichlet`)

### Optimizer

- [ ] Add general interfaces to optax

### 2D/3D FEM

- [ ] Add interface to [JAX-FEM](https://github.com/deepmodeling/jax-fem) for 2D/3D cases

### Nanotubes

- [ ] OPs for cylindrical coordinates

## Documentation

- [x] Add usage example notebook for 1D EDL (extend `examples/00.1d_edl/run.ipynb`)
- [x] Add API docstrings to `ElectricalDoubleLayer`, `Electrode`, `Electrolyte`, `Solver`
- [x] Add API docstrings to isotherm module (`BaseIsotherm`, `LangmuirIsotherm`, `FrumkinIsotherm`)

## Unit Tests

- [ ] Add unit tests for the PBE residual and Newton solver

## Chores

- [x] Set up CI from GitHub workflow files (https://docs.gitlab.com/ci/migration/github_actions/)
- [ ] Check GPU compatibility for JAX solver
