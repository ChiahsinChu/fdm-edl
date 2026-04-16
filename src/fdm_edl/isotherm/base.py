# SPDX-License-Identifier: GPL-3.0-or-later
from abc import ABC, abstractmethod
from typing import Callable

import jax
import optax
import unxt
from jax import numpy as jnp

from ..utils import constants
from ..utils.output_def import IsothermStatus
from ..utils.unit_conversion import check_data_type


class BaseIsotherm(ABC):
    """Abstract base class for adsorption isotherm models.

    Solves for the equilibrium surface coverage :math:`\\theta` at a given
    electrode potential by minimising the isotherm residual with an Adam
    optimiser (via *optax*).

    Parameters
    ----------
    n_et : float
        Number of electrons transferred in the adsorption reaction.
    temperature : unxt.Quantity
        Absolute temperature (must be convertible to ``K``).
    theta_max : float, optional
        Maximum fractional surface coverage (default: ``1.0``).
    **kwargs
        Additional keyword arguments forwarded to subclasses.

    Attributes
    ----------
    status : IsothermStatus or None
        Result container populated after calling :meth:`compute`.
    """

    def __init__(
        self,
        n_et: float,
        temperature: unxt.Quantity,
        theta_max: float = 1.0,
        **kwargs,
    ):
        self.n_et = n_et
        self.temperature = temperature
        self.theta_max = theta_max

        check_data_type(temperature, "temperature")

        self._temperature = temperature.to("K").value

        self.status: IsothermStatus = None

    def compute(
        self,
        phi: unxt.Quantity,
        *,
        max_iter: int = 500,
        lr: float = 0.05,
        tol: float = 1e-10,
        x0: float = 0.0,
    ) -> IsothermStatus:
        """Compute equilibrium surface coverage for the given potential(s).

        Parameters
        ----------
        phi : unxt.Quantity
            Electrode potential (scalar or 1-D array, convertible to ``V``).
        max_iter : int, optional
            Maximum number of optimiser iterations (default: ``500``).
        lr : float, optional
            Adam learning rate (default: ``0.05``).
        tol : float, optional
            Squared-residual convergence tolerance (default: ``1e-10``).
        x0 : float, optional
            Initial value for the unconstrained optimisation variable
            (default: ``0.0``).

        Returns
        -------
        IsothermStatus
            Dataclass containing the potential, coverage, temperature,
            and lateral-interaction energy.
        """
        all_phi = phi.to("V").value

        if len(all_phi.shape) == 0:
            # single value
            all_theta = self._compute(all_phi, max_iter=max_iter, lr=lr, tol=tol, x0=x0)
        elif len(all_phi.shape) == 1:
            # array
            all_theta = []
            for _phi in all_phi:
                theta = self._compute(_phi, max_iter=max_iter, lr=lr, tol=tol, x0=x0)
                all_theta.append(theta)
            all_theta = jnp.array(all_theta)
        else:
            raise ValueError("Unsupported shape for phi")

        self.status = IsothermStatus(
            phi=phi,
            coverage=all_theta,
            temperature=self.temperature,
            n_et=self.n_et,
            coverage_max=self.theta_max,
            lateral_interaction=self.lateral_interaction(all_theta)
            * (constants.MOLAR_GAS_CONSTANT * self.temperature),
        )

        return self.status

    def _compute(
        self,
        _phi: float,
        *,
        max_iter: int = 500,
        lr: float = 0.05,
        tol: float = 1e-10,
        x0: float = 0.0,
    ) -> float | jax.Array:
        """Solve the isotherm for a single scalar potential value.

        Uses a sigmoid reparametrisation to keep :math:`\\theta` in
        :math:`(0, \\theta_{\\max})` and minimises the squared residual
        with an Adam optimiser inside a ``jax.lax.while_loop``.

        Parameters
        ----------
        _phi : float
            Electrode potential in volts (dimensionless scalar).
        max_iter : int, optional
            Maximum optimiser iterations.
        lr : float, optional
            Adam learning rate.
        tol : float, optional
            Squared-residual convergence tolerance.
        x0 : float, optional
            Initial unconstrained variable value.

        Returns
        -------
        float or jax.Array
            Equilibrium surface coverage :math:`\\theta`.
        """
        residual = self._residual_fn(_phi)

        eps = 1e-12

        def x_to_theta(x):
            # theta in (0, theta_max), avoid hitting endpoints exactly
            s = jax.nn.sigmoid(x)
            return eps + (self.theta_max - 2 * eps) * s

        def loss_fn(x):
            th = x_to_theta(x)
            r = residual(th)
            return r * r

        opt = optax.adam(lr)
        opt_state = opt.init(x0)

        x = jnp.asarray(x0)

        grad_fn = jax.grad(loss_fn)

        def body(state):
            x, opt_state, _prev_loss, i = state
            g = grad_fn(x)
            updates, opt_state = opt.update(g, opt_state, x)
            x = optax.apply_updates(x, updates)
            cur_loss = loss_fn(x)
            return x, opt_state, cur_loss, i + 1

        def cond(state):
            x, opt_state, cur_loss, i = state
            return jnp.logical_and(cur_loss > tol, i < max_iter)

        init_loss = loss_fn(x)
        x, opt_state, final_loss, _ = jax.lax.while_loop(
            cond, body, (x, opt_state, init_loss, 0)
        )

        theta = x_to_theta(x)
        return theta

    def _residual_fn(self, phi: float) -> Callable[[float], float | jax.Array]:
        """Build the isotherm residual function for a given potential.

        The residual is

        .. math::

            r(\\theta) = \\ln\\!\\left(\\frac{\\theta}{\\theta_{\\max} - \\theta}\\right)
            + g(\\theta) - \\frac{n_{\\mathrm{et}} F}{RT}\\,\\phi

        where :math:`g(\\theta)` is the lateral-interaction term.

        Parameters
        ----------
        phi : float
            Electrode potential in volts (dimensionless scalar).

        Returns
        -------
        Callable[[float], float | jax.Array]
            A function mapping coverage :math:`\\theta` to the residual.
        """
        rhs_coeff = (
            (
                (self.n_et * constants.FARADAY_CONSTANT)
                / (constants.MOLAR_GAS_CONSTANT * self.temperature)
            )
            .to("1/V")
            .value
        )

        def residual_fn(theta: float) -> float | jax.Array:
            lhs = jnp.log(theta / (self.theta_max - theta)) + self.lateral_interaction(
                theta
            )
            rhs = -rhs_coeff * phi
            return lhs - rhs

        return residual_fn

    @abstractmethod
    def lateral_interaction(self, theta: float | jax.Array) -> float | jax.Array:
        """Return the dimensionless lateral-interaction term :math:`g(\\theta)`.

        Subclasses must implement this method.  The value is added to
        the logarithmic term in the isotherm residual.

        Parameters
        ----------
        theta : float or jax.Array
            Current surface coverage.

        Returns
        -------
        float or jax.Array
            Dimensionless lateral-interaction contribution.
        """
        ...
