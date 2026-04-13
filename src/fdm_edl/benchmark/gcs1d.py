# SPDX-License-Identifier: GPL-3.0-or-later
"""Analytical Gouy-Chapman-Stern (GCS) solution for a z:z electrolyte in 1D.

...
"""

from __future__ import annotations

import jax
import quaxed as qjax
import quaxed.numpy as jnp
import unxt
from astropy.units import cds
from scipy.optimize import brentq

from ..models.base import boltzmann_factor
from ..utils import constants
from ..utils.output import EDLStatus
from .pb1d import NonLinearPoissonBoltzmann


class GCSModel(NonLinearPoissonBoltzmann):
    r"""Analytical GCS solution for a symmetric binary (z:z) electrolyte.

    The electrode potential :math:`\phi_0` is connected to the OHP
    potential :math:`\phi_d` through the Stern capacitor:

    .. math::

        \sigma = \frac{\varepsilon_s}{d}(\phi_0 - \phi_d)

    and also to the diffuse-layer charge via the Grahame relation:

    .. math::

        \sigma = \sqrt{8\,\varepsilon\,k_BT\,c_0\,N_A}\;
                 \sinh\!\left(\frac{z\,e\,\phi_d}{2\,k_BT}\right)

    Equating these gives an implicit equation for :math:`\phi_d`, which
    is solved numerically (Brent).  The diffuse-layer potential profile
    is then the non-linear PB closed form:

    .. math::

        \phi(x) = \frac{4\,k_BT}{ze}\,\mathrm{arctanh}\!\left[
            \tanh\!\left(\frac{ze\,\phi_d}{4\,k_BT}\right)
            \exp\!\left(-\frac{x}{\lambda_D}\right)\right]

    Parameters
    ----------
    edl_obj : ElectricalDoubleLayer
        Configured EDL system.
    d_ohp : unxt.Quantity, optional
        Distance from electrode surface to the outer Helmholtz plane
        (default: 5 Å).
    eps_ohp : float, optional
        Relative permittivity inside the Stern (compact) layer
        (default: 6.0).
    """

    def __init__(
        self,
        edl_obj,
        d_ohp: unxt.Quantity = unxt.Quantity(5.0, "Angstrom"),
        eps_ohp: float = 6.0,
    ):
        super().__init__(edl_obj)
        self.d_ohp = d_ohp
        self.eps_ohp = eps_ohp

    @staticmethod
    @jax.jit
    def compute_phi_helmholtz(x, sigma, eps_ohp, phi_0):
        """Linear potential profile inside the Stern layer.

        Parameters
        ----------
        x : unxt.Quantity
            Distance from the electrode surface.
        sigma : unxt.Quantity
            Surface charge density.
        eps_ohp : float
            Relative permittivity in the Stern layer.
        phi_0 : unxt.Quantity
            Electrode potential.

        Returns
        -------
        unxt.Quantity
            Potential at position *x* within the Stern layer.
        """
        efield = sigma / (eps_ohp * constants.VACUUM_PERMITTIVITY)
        # 'e m / (Angstrom F)' (electrical potential) and '' (dimensionless)
        return phi_0 - efield * x

    def compute(
        self,
        x: unxt.Quantity,
        phi_0: unxt.Quantity,
    ):
        r"""Evaluate the full GCS (Stern + diffuse-layer) solution.

        Solves for the OHP potential :math:`\phi_d` by matching the
        Stern-layer and Gouy-Chapman charge densities, then computes
        potential, electric field, and ion concentration profiles.
        Results are stored in ``self.edl_status``.

        Parameters
        ----------
        x : unxt.Quantity
            Spatial coordinates at which to evaluate the solution.
        phi_0 : unxt.Quantity
            Electrode potential at x = 0.
        """
        _x = x.to("angstrom")
        _phi_0 = phi_0.to("V")
        debye_length = self.edl_obj.electrolyte.debye_length.to("angstrom")
        beta = self.beta

        # since no free charges exist in the Stern layer
        # solve phi_ohp based on sigma_h = sigma_gc
        phi_0_value = _phi_0.value

        def sigma_residual(phi_ohp: float) -> float:
            sigma_h = (
                (phi_0_value - phi_ohp)
                * self.eps_ohp
                * (constants.VACUUM_PERMITTIVITY / self.d_ohp)
                .to(cds.e / unxt.unit("V*angstrom^2"))
                .value
            )
            sigma_gc = (
                self.phi0_to_sigma(unxt.Quantity(phi_ohp, "V"))
                .to(cds.e / unxt.unit("angstrom^2"))
                .value
            )
            sigma_ohp = sigma_h - sigma_gc
            return sigma_ohp

        if abs(phi_0_value) < 1e-15:
            phi_ohp = unxt.Quantity(0.0, "V")
        else:
            a, b = (0.0, phi_0_value) if phi_0_value > 0 else (phi_0_value, 0.0)
            phi_ohp = unxt.Quantity(brentq(sigma_residual, a, b, xtol=1e-14), "V")
        # print(phi_ohp)
        # print(sigma_residual(phi_ohp.value))
        sigma = self.phi0_to_sigma(phi_ohp).to(cds.e / unxt.unit("angstrom^2"))

        # calculate phi profile
        phi = unxt.Quantity(jnp.zeros_like(_x).value, "V")
        _phi = self.compute_phi(
            _x - self.d_ohp.to("angstrom"),
            phi_ohp,
            debye_length,
            beta,
            self.valency,
        )
        mask_gc = _x > self.d_ohp.to("angstrom")
        phi = phi.at[mask_gc].set(unxt.Quantity(_phi.value, unit="V")[mask_gc])
        mask_helmholtz = ~mask_gc
        phi = phi.at[mask_helmholtz].set(
            self.compute_phi_helmholtz(
                _x[mask_helmholtz],
                sigma,
                self.eps_ohp,
                _phi_0,
            )
        )

        # calculate efield profile
        efield = unxt.Quantity(jnp.zeros_like(_x).value, "V/angstrom")
        efield = efield.at[mask_gc].set(
            -qjax.vmap(
                qjax.grad(self.compute_phi), in_axes=(0, None, None, None, None)
            )(
                _x[mask_gc] - self.d_ohp.to("angstrom"),
                phi_ohp,
                debye_length,
                beta,
                self.valency,
            )
        )
        efield = efield.at[mask_helmholtz].set(
            sigma / (self.eps_ohp * constants.VACUUM_PERMITTIVITY)
        )
        # set the ion_conc for x < d_ohp as zero since it's in the Stern layer where ions are absent
        self.edl_status = EDLStatus(
            coordinate=x.to("angstrom"),
            sigma=sigma,
            phi=phi,
            efield=efield.to("V/Angstrom"),
            rho=None,
            ion_conc={
                name: jnp.where(
                    mask_gc,
                    ion.molar_conc
                    * boltzmann_factor(
                        phi=phi,
                        temperature=self.edl_obj.temperature,
                        charge=ion.charge,
                    ),
                    0.0 * ion.molar_conc.unit,  # zero concentration in Stern layer
                )
                for name, ion in self.edl_obj.electrolyte.ions.items()
            },
        )
