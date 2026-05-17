# Getting Started

For a 1-1 electrolyte with a simple Boltzmann distribution, you can initialize the solver with a minimal parameter dictionary:

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

phi_0 = unxt.Quantity(0.1, "V")

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

- `coordinate`: coordinate array
- `sigma`: surface charge density at the electrode (if applicable)
- `phi`: electric potential
- `efield`: electric field
- `rho`: charge density
- `ion_conc`: ionic concentrations
- `epsilon_r`: relative permittivity
