# FDM-EDL

[![codecov](https://codecov.io/gh/ChiahsinChu/fdm-edl/graph/badge.svg?token=CdPAE72ak3)](https://codecov.io/gh/ChiahsinChu/fdm-edl)

Finite-difference electrical double layer simulations with JAX and unit-aware inputs.

`fdm_edl` solves continuum electrical double layer models from JSON/YAML-style parameter sets or Python dictionaries. The current package is focused on 1D problems, with modular charge-density models, dielectric-response models, boundary conditions, and nonlinear solvers.

## Current Scope

- 1D EDL solves through `fdm_edl.api.ElectricalDoubleLayer`
- JAX-backed residual evaluation and nonlinear solves
- Unit-aware inputs and outputs via `unxt.Quantity`
- Configurable electrolyte, solvent, charge model, solver, and gradient operator
- Analytical benchmark models for Gouy-Chapman-Stern and Poisson-Boltzmann cases

## Features

- Charge models: `boltzmann`, `bikerman`
- Solvent dielectric models: `uniform`, `langevin`, `booth`
- Boundary conditions: `ConstP`, `ConstQ`, `Symmetric`, `Stern`
- Solver methods: `newton`, `bicgstab`, `cg`, `gmres`
- Gradient operators:
  - `finite_difference` or `finite_volume`
  - `cartesian` or `axisymmetric`
- Built-in benchmarks: `LinearPoissonBoltzmann`, `NonLinearPoissonBoltzmann`, `GCSModel`
- Input loading from Python dictionaries, JSON, and YAML files

## Installation

Requires Python 3.11 to 3.13.

```bash
conda create -n fdm-edl python=3.11 -y
conda activate fdm-edl

git clone https://github.com/ChiahsinChu/fdm-edl.git
cd fdm-edl
pip install .
```

Development install:

```bash
pip install -e ".[test,docs]"
```

If you want to load YAML parameter files, ensure `PyYAML` is installed in your environment.

## Parameter File Format

You can initialize the solver from a dictionary or from a `.json`, `.yml`, or `.yaml` file.

Example:

```json
{
  "unit": "metal",
  "temperature": 298.15,
  "electrolyte": {
    "solvent": {
      "type": "booth",
      "epsilon_r_0": 78.4,
      "epsilon_r_inf": 1.78
    },
    "ions": {
      "Na": {
        "molar_conc": 0.01,
        "charge": 1.0,
        "radius": 3.6
      },
      "Cl": {
        "molar_conc": 0.01,
        "charge": -1.0,
        "radius": 3.3
      }
    }
  },
  "model": {
    "type": "bikerman"
  },
  "solver": {
    "method": "newton",
    "max_iter": 500,
    "atol_var": 1e-6
  },
  "grad_op": {
    "type": "finite_volume",
    "coordinate_system": "cartesian"
  }
}
```

## Public API

Main imports:

```python
from fdm_edl.api import (
    ElectricalDoubleLayer,
    Electrolyte,
    ConstP,
    ConstQ,
    Symmetric,
    Stern,
)
```

Benchmarks:

```python
from fdm_edl.benchmark import (
    GCSModel,
    LinearPoissonBoltzmann,
    NonLinearPoissonBoltzmann,
)
```

## Testing

```bash
pytest tests/
```

The test suite covers:

- gradient operators
- nonlinear Poisson-Boltzmann regression
- solver backends
- dielectric-response models

## License

This project is licensed under the LGPL-3.0-or-later license. See [LICENSE](LICENSE).
