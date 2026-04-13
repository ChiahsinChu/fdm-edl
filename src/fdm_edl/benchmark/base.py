# SPDX-License-Identifier: GPL-3.0-or-later
import quaxed.numpy as jnp
import unxt


# @jax.jit
def linear_exponent(x: unxt.Quantity, debye_length: unxt.Quantity) -> unxt.Quantity:
    """Dimensionless exponential decay factor exp(-x / λ_D)."""
    return jnp.exp(-(x / debye_length).to(""))
