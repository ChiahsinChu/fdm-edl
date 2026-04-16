# SPDX-License-Identifier: GPL-3.0-or-later
import jax
from jax import numpy as jnp


def langevin_function(x: jax.Array) -> jax.Array:
    """Langevin function.

    Args:
        x: Input array.

    Returns:
        Langevin function applied to the input array.
    """
    return (jnp.cosh(x) / jnp.sinh(x)) - 1 / x
