# SPDX-License-Identifier: GPL-3.0-or-later
import quaxed.numpy as jnp
import unxt

from .. import _constants


# @jax.jit
def boltzmann_factor(
    phi: unxt.Quantity,
    temperature: unxt.Quantity,
    valency: int = 1,
):
    """Compute the Boltzmann factor exp(-z e φ / k_B T).

    Parameters
    ----------
    phi : unxt.Quantity
        Electrostatic potential.
    temperature : unxt.Quantity
        Absolute temperature.
    valency : int, optional
        Ion valency *z* (default: 1).

    Returns
    -------
    unxt.Quantity
        Dimensionless Boltzmann factor at each grid node.
    """
    beta = 1.0 / (_constants.BOLTZMANN_CONSTANT * temperature).to("eV")
    return jnp.exp((-_constants.ELEMENTARY_CHARGE * phi * beta * valency).to(""))


# @jax.jit
def linear_exponent(x: unxt.Quantity, debye_length: unxt.Quantity) -> unxt.Quantity:
    """Dimensionless exponential decay factor exp(-x / λ_D)."""
    return jnp.exp(-(x / debye_length).to(""))
