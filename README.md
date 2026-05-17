# FDM-EDL

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

## Quick Start

```python
from jax import numpy as jnp
import unxt

from fdm_edl.api import ConstP, ElectricalDoubleLayer

params = {
    "unit": "metal",
    "temperature": 298.15,
    "electrolyte": {
        "ions": {
            "Na": {"molar_conc": 0.01, "charge": 1.0},
            "Cl": {"molar_conc": 0.01, "charge": -1.0},
        },
    },
    "model": {"type": "boltzmann"},
    "solver": {"method": "newton", "max_iter": 500, "atol_var": 1e-6},
}

edl = ElectricalDoubleLayer(params)
debye_length = edl.electrolyte.debye_length

n_grid = 500
x = unxt.Quantity(
    jnp.linspace(0.0, debye_length.to("nm").value * 10.0, n_grid),
    unit="nm",
)

phi_0 = unxt.Quantity(0.025, "V")

bcs = ()
bcs += ConstP(phi=phi_0)([0])
bcs += ConstP(phi=unxt.Quantity(0.0, "V"))([n_grid - 1])

edl.compute(x, bcs)

result = edl.result
phi = result.phi
sigma = result.sigma
rho = result.rho
ion_conc = result.ion_conc
```

`ElectricalDoubleLayer.compute()` stores results in `edl.result`, an `EDLStatus` object with:

- `coordinate`
- `sigma`
- `phi`
- `efield`
- `rho`
- `ion_conc`
- `epsilon_r`

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

Supported top-level keys:

| Key                   | Meaning                                                                                                                                              |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `unit`                | [Lammps-type](https://docs.lammps.org/units.html) unit system for input parameters. Defaults to `metal`.                                             |
| `temperature`         | Absolute temperature. Required.                                                                                                                      |
| `dim`                 | Model dimension. The current solver implementation supports `dim = 1`.                                                                               |
| `electrode`           | Electrode settings. Present for API completeness; currently minimal.                                                                                 |
| `electrolyte.ions`    | Ion dictionary keyed by species name. Each ion supports `charge`, `molar_conc`, and optional `radius`.                                               |
| `electrolyte.solvent` | Solvent dielectric model. `type` can be `uniform`, `booth`, or `langevin`.                                                                           |
| `model`               | Charge-density model. `type` can be `boltzmann` or `bikerman`.                                                                                       |
| `solver`              | Nonlinear solver settings such as `method`, `max_iter`, `atol_var`, and `atol_res`.                                                                  |
| `grad_op`             | Discrete gradient operator settings. `type` can be `finite_difference` or `finite_volume`; `coordinate_system` can be `cartesian` or `axisymmetric`. |

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

## Repository Layout

```text
src/fdm_edl/
├── api/         # User-facing EDL objects and boundary-condition helpers
├── benchmark/   # Analytical and semi-analytical reference models
├── isotherm/    # Surface isotherm models
├── models/      # Charge-density and solvent dielectric-response models
├── op/          # Finite-difference / finite-volume discrete operators
├── solver/      # Newton and SciPy-based nonlinear solvers
└── utils/       # I/O, constants, units, and output dataclasses
```

## Examples

- `examples/00.1D-PB_analytical/`: linear and nonlinear Poisson-Boltzmann notebooks
- `examples/01.bikerman/`: Bikerman model example with Stern boundary conditions and field-dependent dielectric response

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
