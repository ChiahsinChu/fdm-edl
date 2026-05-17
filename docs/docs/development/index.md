# Development

The repository layout is shown below. The core EDL solver and model implementations are in `src/fdm_edl/`, which is organized into subpackages for API objects, physical models, numerical solvers, and utilities.

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

The users are expected to use the objects and functions exposed in `fdm_edl.api` in a standarized EDL modelling.
Particularly, `unxt.Quantity` objects are expected to be only used in the high-level API. The user-defined physical quantities will be converted to the internal unit system (default: `metal`) and passed down to the model and solver layers as unitless floats or arrays. The API will convert the unitless outputs back to `unxt.Quantity` objects in the original input units for user convenience.
