# FDM-EDL

A Python package for solving the Poisson-Boltzmann equation in electrical double layers (EDLs) using the finite difference method (FDM). Built on [JAX](https://github.com/jax-ml/jax) for automatic differentiation and GPU acceleration, with full SI-unit support via [unxt](https://github.com/GalacticDynamics/unxt).

## Features

- **Poisson-Boltzmann solver** for 1D EDL simulations with multi-ion electrolytes
- **JAX-based numerics** — autodiff Jacobians and potential GPU acceleration
- **SI units throughout** using `unxt.Quantity` for dimensional consistency
- **Flexible boundary conditions** — Dirichlet, Neumann, Robin, and Periodic BCs
- **Multiple solvers** — Newton's method (JAX autodiff Jacobian) and Optax gradient-based optimizers (Adam, SGD, RMSProp)
- **Nonuniform grid support** — 1D Laplacian operator handles variable node spacing
- **Analytical benchmarks** — built-in linear and non-linear PB solutions for validation
- **JSON/YAML configuration** — define electrolyte composition and solver settings in a config file

## Installation

Requires Python ≥ 3.11.

```bash
conda create -n fdm-edl python=3.11 -y
conda activate fdm-edl

git clone https://jugit.fz-juelich.de/ZhuJia-Xin/fdm-edl.git
cd fdm-edl
pip install .
```

For development (tests + docs):

```bash
pip install -e ".[test,docs]"
```

## Quick Start

```python
from fdm_edl.edl import ElectricalDoubleLayer
from fdm_edl.bc import DirichletBC
import unxt
import quaxed.numpy as jnp

# 1. Initialize system from a JSON config
edl = ElectricalDoubleLayer("input.json")

# 2. Create a 1D grid
n_grid = 500
x = unxt.Quantity(jnp.linspace(0, 50, n_grid), "nm")

# 3. Define boundary conditions
bcs = [
    DirichletBC([0], unxt.Quantity(0.1, "V")),  # electrode potential
    DirichletBC([n_grid - 1], unxt.Quantity(0.0, "V")),  # bulk solution
]

# 4. Solve
edl.compute(x, bcs)

# 5. Post-process
phi = edl.result.solution  # potential profile
ion_conc = edl.get_ion_concentration_profiles()  # dict of ion concentrations
```

## Input File Format

Parameters are loaded from a JSON or YAML file:

```json
{
  "temperature": { "value": 298, "unit": "K" },
  "electrode": {},
  "electrolyte": {
    "epsilon_r": 78.5,
    "ions": {
      "Na": {
        "molar_conc": { "value": 0.01, "unit": "mol/L" },
        "charge": { "value": 1.0, "unit": "e" }
      },
      "Cl": {
        "molar_conc": { "value": 0.01, "unit": "mol/L" },
        "charge": { "value": -1.0, "unit": "e" }
      }
    }
  },
  "solver": { "method": "newton", "max_iter": 30, "tol": 1e-6 }
}
```

| Field                            | Description                                          |
| -------------------------------- | ---------------------------------------------------- |
| `temperature`                    | System temperature with unit                         |
| `electrolyte.epsilon_r`          | Relative permittivity of the solvent (default: 78.5) |
| `electrolyte.ions`               | Dict of ions, each with `molar_conc` and `charge`    |
| `solver.method`                  | `"newton"`, `"adam"`, `"sgd"`, or `"rmsprop"`        |
| `solver.max_iter` / `solver.tol` | Convergence controls                                 |

## Package Structure

```
src/fdm_edl/
├── edl/          # Main API: ElectricalDoubleLayer, Electrode, Electrolyte, Ion
├── bc/           # Boundary conditions: Dirichlet, Neumann, Robin, Periodic
├── operators/    # FDM operators: LaplacianOperator1D (3-point stencil)
├── solver/       # Solvers: NewtonSolver, OptaxSolver (factory registry)
├── mesh/         # Mesh generation: LineMesh (via Gmsh)
├── test/         # Analytical solutions: LinearPB, NonLinearPB
└── utils/        # I/O helpers, unit conversion
```

## Examples

See [`examples/00.1D-PB_analytical/`](examples/00.1D-PB_analytical/) for Jupyter notebooks validating the solver against analytical Poisson-Boltzmann solutions:

- **Linear PB** — small-potential (Debye-Hückel) regime
- **Non-linear PB** — full non-linear solution for 1:1 electrolytes

## Testing

```bash
pytest tests/
```

## License

LGPL-3.0-or-later
