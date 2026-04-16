# SPDX-License-Identifier: GPL-3.0-or-later
import jax
import unxt

from ..utils import constants
from ..utils.unit_conversion import check_data_type
from .base import BaseIsotherm


class FrumkinIsotherm(BaseIsotherm):
    """Frumkin isotherm with coverage-dependent lateral interactions.

    The dimensionless lateral-interaction term is

    .. math::

        g(\\theta) = \\frac{\\omega}{RT}\\,(\\alpha\\,\\theta + \\beta)\\,\\theta

    where :math:`\\omega` is the lateral-interaction energy,
    :math:`\\alpha` and :math:`\\beta` are shape parameters.

    Parameters
    ----------
    n_et : float
        Number of electrons transferred.
    temperature : unxt.Quantity
        Absolute temperature.
    omega : unxt.Quantity
        Lateral-interaction energy per mole (e.g. ``kJ / mol``).
    theta_max : float, optional
        Maximum fractional surface coverage (default: ``1.0``).
    alpha : float, optional
        Quadratic shape parameter (default: ``0.0``).
    beta : float, optional
        Linear shape parameter (default: ``1.0``).
    """

    def __init__(
        self,
        n_et: float,
        temperature: unxt.Quantity,
        omega: unxt.Quantity,
        theta_max: float = 1.0,
        alpha: float = 0.0,
        beta: float = 1.0,
    ):
        super().__init__(n_et, temperature, theta_max)
        self.omega = omega
        self.alpha = alpha
        self.beta = beta

        check_data_type(self.omega, "chemical potential")

    def lateral_interaction(self, theta: float | jax.Array) -> float | jax.Array:
        """Compute the Frumkin lateral-interaction term.

        Parameters
        ----------
        theta : float or jax.Array
            Current surface coverage.

        Returns
        -------
        float or jax.Array
            Dimensionless interaction :math:`(\\omega / RT)(\\alpha\\theta + \\beta)\\theta`.
        """
        coeff = (
            (self.omega / (constants.MOLAR_GAS_CONSTANT * self.temperature))
            .to("")
            .value
        )
        return coeff * (self.alpha * theta + self.beta) * theta
