# SPDX-License-Identifier: GPL-3.0-or-later
import jax

from .base import BaseIsotherm


class LangmuirIsotherm(BaseIsotherm):
    """Langmuir isotherm (no lateral interactions).

    The lateral-interaction term is identically zero, reducing the
    general isotherm to the classical Langmuir equation.
    """

    def lateral_interaction(self, theta: float | jax.Array) -> float | jax.Array:
        """Return zero lateral interaction.

        Parameters
        ----------
        theta : float or jax.Array
            Surface coverage (unused; kept for API consistency).

        Returns
        -------
        float or jax.Array
            Always ``0.0 * theta`` (preserves JAX tracing).
        """
        return 0.0 * theta
