---
status: draft
author: Jia-Xin Zhu, AI Agent
last_updated: 2026-03-17
---

# Development Plans

## Features

### Done

- [x] `ElectricalDoubleLayer` base class with `compute()` and `plot()` methods (task.000)
- [x] SI-unit multi-ion Poisson-Boltzmann solver
- [x] Solvent dielectric constant `epsilon_r` (default 78.5 for water)
- [x] Load parameters from `dict`, JSON, or YAML file
- [x] Electroneutrality check on electrolyte input

### Pending

- [ ] Allow more flexible / automated grid generation (e.g., non-uniform, adaptive spacing)
- [ ] Compute analytical properties (e.g., Debye length) without solving the full PDE
- [ ] Add physical units to output results and plots (e.g., via [`unxt`](https://github.com/GalacticDynamics/unxt))

## Documentation

- [ ] Add usage example notebook for 1D EDL (extend `examples/00.1d_edl/run.ipynb`)
- [ ] Add API docstrings to `ElectricalDoubleLayer`, `Electrode`, `Electrolyte`, `Solver`

## Chores

- [ ] Check GPU compatibility for JAX solver
- [ ] Add unit tests for the PBE residual and Newton solver
