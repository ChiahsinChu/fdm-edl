# SPDX-License-Identifier: GPL-3.0-or-later
"""Stern layer boundary condition for the Gouy-Chapman-Stern model.

Encodes the compact (Stern) layer as a Robin boundary condition at the
Outer Helmholtz Plane (OHP), so the computational domain covers only
the diffuse layer.
"""

from __future__ import annotations

import jax.numpy as raw_jnp
import unxt

from .base import BoundaryCondition


class SternLayerBC(BoundaryCondition):
    r"""Robin BC encoding Stern-layer capacitance at the OHP.

    The Stern layer of thickness *d* and permittivity
    :math:`\varepsilon_s` relates the electrode potential
    :math:`\phi_0` to the OHP potential :math:`\phi_d` via:

    .. math::

        \varepsilon_s \frac{\phi_0 - \phi_d}{d}
          = \varepsilon_{bulk}\;\frac{\partial\phi}{\partial x}\bigg|_{OHP}

    This is cast into the Robin form
    :math:`\alpha\,\phi + \beta\,\partial\phi/\partial n = g` with:

    * :math:`\alpha = \varepsilon_s / d`
    * :math:`\beta  = -\varepsilon_{bulk}`
    * :math:`g      = \varepsilon_s\,\phi_0 / d`

    The sign convention assumes the outward normal at the OHP points
    *away* from the electrode (into the diffuse layer), matching the
    one-sided finite-difference stencil ``(φ[i] − φ[j]) / (x[i] − x[j])``
    used by :class:`~fdm_edl.bc.RobinBC`.

    Parameters
    ----------
    node_indices : array-like of int
        Grid index (or indices) at the OHP.
    electrode_potential : unxt.Quantity
        Applied electrode potential :math:`\phi_0`.
    stern_thickness : unxt.Quantity
        Stern-layer thickness *d*.
    stern_permittivity : unxt.Quantity
        Permittivity of the Stern layer :math:`\varepsilon_s`
        (absolute, in F/m).
    bulk_permittivity : unxt.Quantity
        Permittivity of the bulk electrolyte :math:`\varepsilon_{bulk}`
        (absolute, in F/m).
    neighbor_indices : array-like of int
        Interior neighbor index used for the one-sided FD flux estimate.
    """

    def __init__(
        self,
        node_indices,
        electrode_potential: unxt.Quantity,
        stern_thickness: unxt.Quantity,
        stern_permittivity: unxt.Quantity,
        bulk_permittivity: unxt.Quantity,
        neighbor_indices,
    ):
        super().__init__(node_indices)
        self.electrode_potential = electrode_potential
        self.stern_thickness = stern_thickness
        self.stern_permittivity = stern_permittivity
        self.bulk_permittivity = bulk_permittivity
        self.neighbor_indices = raw_jnp.asarray(neighbor_indices)

    def apply_residual(self, residual, phi, coordinates):
        """Enforce the Stern-layer Robin condition at OHP nodes.

        Parameters
        ----------
        residual : unxt.Quantity, shape (n_grid,)
            Physics residual at every grid node.
        phi : unxt.Quantity, shape (n_grid,)
            Current potential at every grid node.
        coordinates : unxt.Quantity
            Grid coordinates (1-D).

        Returns
        -------
        unxt.Quantity, shape (n_grid,)
            Residual with the Stern BC applied at ``self.node_indices``.
        """
        x_bc = coordinates[self.node_indices].to("m")
        x_nb = coordinates[self.neighbor_indices].to("m")
        dx = x_bc - x_nb

        phi_bc = phi[self.node_indices].to("V")
        phi_nb = phi[self.neighbor_indices].to("V")
        fd_deriv = (phi_bc - phi_nb) / dx  # ∂φ/∂n (outward), V/m

        # Robin:  (ε_s / d) φ  +  (-ε_bulk) ∂φ/∂n  =  (ε_s / d) φ_0
        alpha = self.stern_permittivity.to("F/m") / self.stern_thickness.to("m")
        beta = -self.bulk_permittivity.to("F/m")
        g = alpha * self.electrode_potential.to("V")

        bc_value = (alpha * phi_bc + beta * fd_deriv - g).value
        bc_residual = unxt.Quantity(bc_value, unit=residual.unit)
        return residual.at[self.node_indices].set(bc_residual)

    def compute_violation(self, phi, coordinates):
        """Return the Stern-layer Robin violation at OHP nodes."""
        x_bc = coordinates[self.node_indices].to("m")
        x_nb = coordinates[self.neighbor_indices].to("m")
        dx = x_bc - x_nb
        phi_bc = phi[self.node_indices].to("V")
        phi_nb = phi[self.neighbor_indices].to("V")
        fd_deriv = (phi_bc - phi_nb) / dx
        alpha = self.stern_permittivity.to("F/m") / self.stern_thickness.to("m")
        beta = -self.bulk_permittivity.to("F/m")
        g = alpha * self.electrode_potential.to("V")
        return alpha * phi_bc + beta * fd_deriv - g

    def apply_initial_guess(self, phi0: unxt.Quantity) -> unxt.Quantity:
        """Seed OHP potential with a reduced electrode potential estimate.

        Uses the Stern capacitor voltage divider as a rough initial guess:

        .. math::

            \phi_{OHP} \approx \phi_0 / 2
        """
        guess = self.electrode_potential / 2.0
        val = unxt.Quantity(
            raw_jnp.full(self.node_indices.shape, guess.to(phi0.unit).value),
            unit=phi0.unit,
        )
        return phi0.at[self.node_indices].set(val)
