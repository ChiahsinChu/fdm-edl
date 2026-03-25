# FDM-EDL

Finite Difference Method for Electrical Double Layer simulations.

A Python/JAX package for solving the Poisson-Boltzmann equation in 1-D, 2-D,
and 3-D using finite-difference methods with modular boundary conditions and
discrete operators.

## Quick Start

```python
from fdm_edl.edl import ElectricalDoubleLayer
from fdm_edl.bc import DirichletBC
import unxt
import quaxed.numpy as jnp

# Load parameters
edl = ElectricalDoubleLayer("input.json")

# Define grid
n_grid = 500
x = unxt.Quantity(jnp.linspace(0.0, 50.0, n_grid), unit="nm")

# Define boundary conditions
bcs = [
    DirichletBC([0], unxt.Quantity(jnp.array([-0.025]), unit="V")),
    DirichletBC([n_grid - 1], unxt.Quantity(jnp.zeros((1,)), unit="V")),
]

# Solve
edl.compute(x, bcs)
```

## Features

- **Multi-dimensional**: Coordinates as `(n_grid, n_dim)` for 1-D/2-D/3-D
- **Modular boundary conditions**: Dirichlet, Neumann, Robin, Periodic
- **External Laplacian operators**: Decoupled discretisation from physics
- **JAX-powered solvers**: Newton-Raphson with automatic Jacobian, Optax optimizers
- **Physical units**: Full unit support via `unxt`
