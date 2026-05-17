# Parameters

## Global Parameters

| Key           | Type    | Default                         | Meaning                                                            |
| ------------- | ------- | ------------------------------- | ------------------------------------------------------------------ |
| `unit`        | string  | `"metal"`                       | Unit system used to interpret numeric input values.                |
| `temperature` | float   | Required                        | Absolute temperature in the selected `unit` system.                |
| `dim`         | integer | `1`                             | Spatial dimension. Current implementation supports only `dim = 1`. |
| `electrode`   | dict    | `{}`                            | Electrode settings.                                                |
| `electrolyte` | dict    | `{}`                            | Electrolyte settings (ions and solvent model).                     |
| `model`       | dict    | `{ "type": "boltzmann" }`       | Ionic charge-density model settings.                               |
| `solver`      | dict    | `{}`                            | Nonlinear solver settings.                                         |
| `grad_op`     | dict    | Auto-selected from solvent type | Gradient/divergence operator settings.                             |

Notes:

- `params` can be passed as a Python dictionary or as a `.json` / `.yaml` file path.
- If `grad_op` is omitted, the operator is auto-selected:
  - `uniform` solvent -> finite-difference (`EuclideanFDOp`)
  - `langevin` and `booth` solvents -> finite-volume (`EuclideanFVOp`)

## Electrode Parameters

`electrode` is currently minimal in the high-level `ElectricalDoubleLayer` setup.

Current behavior: `ElectricalDoubleLayer.set_electrode()` initializes `Electrode(temperature=self.temperature)` and does not yet forward user-provided sub-keys from `params["electrode"]`.

## Electrolyte Parameters

Top-level electrolyte keys:

| Key       | Type | Default                 | Meaning                                          |
| --------- | ---- | ----------------------- | ------------------------------------------------ |
| `ions`    | dict | `{}`                    | Mapping from ion name to ion properties.         |
| `solvent` | dict | `{ "type": "uniform" }` | Solvent dielectric-response model configuration. |

### `electrolyte.ions.<ion_name>`

Each ion entry is converted to an `Ion` object.

| Key          | Type  | Default  | Meaning                                                                                   |
| ------------ | ----- | -------- | ----------------------------------------------------------------------------------------- |
| `charge`     | float | Required | Ionic charge number in the selected unit system (for example `+1`, `-1`, `+2`).           |
| `molar_conc` | float | Required | Bulk concentration in `mol / L`.                                                          |
| `radius`     | float | `0.0`    | Effective ion radius in the selected length unit. Needed for steric models like Bikerman. |

### `electrolyte.solvent`

Common key:

| Key    | Type   | Default     | Meaning                                                                         |
| ------ | ------ | ----------- | ------------------------------------------------------------------------------- |
| `type` | string | `"uniform"` | Solvent dielectric model. Allowed values: `"uniform"`, `"langevin"`, `"booth"`. |

Model-specific solvent keys:

- `type: "uniform"`

  | Key         | Type  | Default | Meaning                         |
  | ----------- | ----- | ------- | ------------------------------- |
  | `epsilon_r` | float | `78.5`  | Constant relative permittivity. |

- `type: "langevin"`

  | Key             | Type  | Default                | Meaning                                                            |
  | --------------- | ----- | ---------------------- | ------------------------------------------------------------------ |
  | `epsilon_r_inf` | float | `1.78`                 | High-frequency (optical) relative permittivity.                    |
  | `mu`            | float | `3.0 debye` equivalent | Molecular dipole moment. If omitted, a water-like default is used. |

- `type: "booth"`

  | Key             | Type  | Default                  | Meaning                                         |
  | --------------- | ----- | ------------------------ | ----------------------------------------------- |
  | `epsilon_r_0`   | float | `78.4`                   | Static relative permittivity.                   |
  | `epsilon_r_inf` | float | `1.78`                   | High-frequency (optical) relative permittivity. |
  | `booth_beta`    | float | `1.41e-8 m/V` equivalent | Booth saturation parameter.                     |

## Charge Model Parameters

| Key    | Type   | Default       | Meaning                                                               |
| ------ | ------ | ------------- | --------------------------------------------------------------------- |
| `type` | string | `"boltzmann"` | Charge model identifier. Allowed values: `"boltzmann"`, `"bikerman"`. |

## Solver Parameters

`solver` keys are forwarded to `BaseSolver(method=...)`, which dispatches to a concrete solver.

| Key           | Type          | Default    | Meaning                                                                      |
| ------------- | ------------- | ---------- | ---------------------------------------------------------------------------- |
| `method`      | string        | `"newton"` | Solver backend. Allowed values: `"newton"`, `"bicgstab"`, `"cg"`, `"gmres"`. |
| `max_iter`    | integer       | `20`       | Maximum outer nonlinear iterations.                                          |
| `max_iter_ls` | integer       | `10`       | Maximum inner linear-solver/backtracking iterations per outer step.          |
| `atol_var`    | float or null | `1e-6`     | Absolute tolerance on potential update.                                      |
| `rtol_var`    | float or null | `null`     | Relative tolerance on potential update.                                      |
| `atol_grad`   | float or null | `null`     | Absolute tolerance on field/gradient update.                                 |
| `rtol_grad`   | float or null | `null`     | Relative tolerance on field/gradient update.                                 |
| `atol_src`    | float or null | `null`     | Absolute tolerance on source-term (charge density) update.                   |
| `rtol_src`    | float or null | `null`     | Relative tolerance on source-term (charge density) update.                   |
| `atol_res`    | float or null | `null`     | Absolute tolerance on residual vector.                                       |
| `rtol_res`    | float or null | `null`     | Relative tolerance on residual vector.                                       |

- `method: "newton"`

  | Key     | Type  | Default | Meaning                                                |
  | ------- | ----- | ------- | ------------------------------------------------------ |
  | `alpha` | float | `1.0`   | Initial line-search step size (used by Newton solver). |

Notes:

- `atol_var`, `atol_grad`, `atol_src`, and `atol_res` are interpreted in the selected `unit` system and converted internally.
- For Krylov-based methods (`bicgstab`, `cg`, `gmres`), extra keys can be passed through to [`jax.scipy.sparse.linalg`](https://docs.jax.dev/en/latest/jax.scipy.html#module-jax.scipy.sparse.linalg) solvers.

## Gradient Operator Parameters

If `grad_op` is provided, it is created with `create_gradient_op(type=..., coordinate_system=..., **kwargs)`.

| Key                 | Type    | Default                    | Meaning                                                                 |
| ------------------- | ------- | -------------------------- | ----------------------------------------------------------------------- |
| `type`              | string  | Auto-selected when omitted | Discretization type. Allowed: `"finite_difference"`, `"finite_volume"`. |
| `coordinate_system` | string  | `"cartesian"`              | Coordinate system. Allowed: `"cartesian"`, `"axisymmetric"`.            |
| `uniform`           | bool    | `true`                     | FD-grid assumption flag (used by finite-difference operators).          |
| `boundary_points`   | integer | `4`                        | Boundary stencil size (`3`, `4`, or `5`).                               |
| `interior_points`   | integer | `3`                        | Interior stencil size (`3` or `5`). Must be `<= boundary_points`.       |

Compatibility rule:

- For non-uniform dielectric solvents (`langevin`, `booth`), finite-difference operators are rejected; use finite-volume.
