# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from typing import TYPE_CHECKING

import unxt

if TYPE_CHECKING:
    from typing import Sequence, Tuple

    import jax

from abc import ABC, abstractmethod

from ..utils import constants
from ..utils.bc import BoundaryCondition
from ..utils.unit_conversion import check_data_type


class TemplateBC(ABC):
    """Base template for boundary-condition coefficient providers.

    Subclasses implement :meth:`make_coeff` to define ``(alpha, beta, gamma)``
    in ``alpha * u + beta * du/dn + gamma = 0``. Calling an instance with
    node indices creates the corresponding :class:`BoundaryCondition` object.
    """

    def __init__(self, **kwargs):
        self.coeff = self.make_coeff(**kwargs)

    @abstractmethod
    def make_coeff(self, **kwargs) -> Tuple[unxt.Quantity | float]: ...

    def __call__(
        self,
        node_indices: jax.Array | Sequence[int],
    ) -> Tuple[BoundaryCondition, ...]:
        """Build a boundary condition tuple for the specified nodes.

        Parameters
        ----------
        node_indices : jax.Array | Sequence[int]
            Indices of the grid nodes where the BC is applied.

        Returns
        -------
        tuple of BoundaryCondition
            A single-element tuple containing the boundary condition.
        """
        return (
            BoundaryCondition(
                self.coeff[0],
                self.coeff[1],
                self.coeff[2],
                node_indices,
            ),
        )


class ConstP(TemplateBC):
    """Constant-potential (Dirichlet) boundary condition.

    Sets ``alpha * u + gamma = 0`` with ``alpha = 1`` and ``gamma = -phi``,
    enforcing ``u = phi`` at the specified nodes.

    Parameters
    ----------
    phi : unxt.Quantity
        Potential at the boundary, relative to the potential of zero charge
        (PZC).
    """

    def __init__(self, phi: unxt.Quantity):
        super().__init__(phi=phi)

    def make_coeff(self, phi) -> Tuple[unxt.Quantity | float]:
        check_data_type(phi, "electrical potential")
        return (1.0, 0.0, -phi)


class ConstQ(TemplateBC):
    """Constant-charge (Neumann) boundary condition.

    Sets ``beta * du/dn = sigma / (epsilon_0 * epsilon_r)`` with ``beta = 1``,
    enforcing a fixed normal electric field corresponding to the specified
    surface charge density via Gauss's law.

    Parameters
    ----------
    sigma : unxt.Quantity
        Surface charge density at the boundary.
    eps_r : float, default 78.5
        Relative permittivity of the electrolyte, used to convert surface
        charge density to electric field via Gauss's law. The default value
        corresponds to bulk water at room temperature.
    """

    def __init__(self, sigma: unxt.Quantity, eps_r: float = 78.5):
        super().__init__(sigma=sigma, eps_r=eps_r)

    def make_coeff(self, sigma, eps_r) -> Tuple[unxt.Quantity | float]:
        check_data_type(sigma, "surface charge density")
        return (0.0, 1.0, -sigma / (constants.VACUUM_PERMITTIVITY * eps_r))


class Symmetric(TemplateBC):
    """Symmetry (zero-flux Neumann) boundary condition.

    Sets ``beta * du/dn = 0`` with ``beta = 1``, enforcing zero normal
    derivative at the specified nodes.
    """

    def __init__(self):
        super().__init__()

    def make_coeff(self) -> Tuple[unxt.Quantity | float]:
        return (0.0, 1.0, unxt.Quantity(0.0, "V"))


class Stern(TemplateBC):
    """Stern-layer (Robin) boundary condition.

    Models the potential drop across a compact Stern layer by setting
    ``alpha = 1`` and ``beta = -(eps_gc / eps_s) * d_s``, which couples the
    surface potential to the normal electric field at the outer Helmholtz
    plane.

    Parameters
    ----------
    phi : unxt.Quantity
        Potential at the boundary, relative to the potential of zero charge
        (PZC).
    eps_gc : float
        Relative permittivity of the diffuse (Gouy-Chapman) layer.
    eps_s : float
        Relative permittivity of the Stern layer.
    d_s : unxt.Quantity
        Thickness of the Stern layer.
    """

    def __init__(
        self,
        phi: unxt.Quantity,
        eps_gc: float,
        eps_s: float,
        d_s: unxt.Quantity,
    ):
        super().__init__(
            phi=phi,
            eps_gc=eps_gc,
            eps_s=eps_s,
            d_s=d_s,
        )

    def make_coeff(self, phi, eps_gc, eps_s, d_s) -> Tuple[unxt.Quantity | float]:
        check_data_type(phi, "electrical potential")
        check_data_type(d_s, "length")
        return (
            1.0,
            -(eps_gc / eps_s) * d_s,
            -phi,
        )


# def make_constp_bc(
#     node_indices: jax.Array | Sequence[int],
#     *,
#     phi: unxt.Quantity,
# ) -> Tuple[BoundaryCondition, ...]:
#     """Create a constant-potential (Dirichlet) boundary condition.

#     Sets ``alpha * u + gamma = 0`` with ``alpha = 1`` and ``gamma = -phi``,
#     enforcing ``u = phi`` at the specified nodes.

#     Parameters
#     ----------
#     node_indices : jax.Array | Sequence[int]
#         Indices of the grid nodes where the BC is applied.
#     phi : unxt.Quantity
#         Potential at the boundary, relative to the potential of zero charge
#         (PZC).

#     Returns
#     -------
#     tuple of BoundaryCondition
#         A single-element tuple containing the Dirichlet BC.
#     """
#     check_data_type(phi, "electrical potential")

#     return (
#         BoundaryCondition(
#             1.0,
#             0.0,
#             -phi,
#             node_indices,
#         ),
#     )


# def make_constq_bc(
#     node_indices: jax.Array | Sequence[int],
#     *,
#     sigma: unxt.Quantity,
#     eps_r: float = 78.5,
# ) -> Tuple[BoundaryCondition, ...]:
#     """Create a constant-charge (Neumann) boundary condition.

#     Sets ``beta * du/dn = sigma / (epsilon_0 * epsilon_r)`` with ``beta = 1``,
#     enforcing a fixed normal electric field corresponding to the specified
#     surface charge density via Gauss's law.

#     Parameters
#     ----------
#     node_indices : jax.Array | Sequence[int]
#         Indices of the grid nodes where the BC is applied.
#     sigma : unxt.Quantity
#         Surface charge density at the boundary.
#     eps_r : float, default 78.5
#         Relative permittivity of the electrolyte, used to convert surface
#         charge density to electric field via Gauss's law.  The default value
#         corresponds to bulk water at room temperature; for other solvents or
#         conditions, the appropriate value should be used.

#     Returns
#     -------
#     tuple of BoundaryCondition
#         A single-element tuple containing the Neumann BC.
#     """
#     check_data_type(sigma, "surface charge density")
#     return (
#         BoundaryCondition(
#             0.0,
#             1.0,
#             -sigma / (constants.VACUUM_PERMITTIVITY * eps_r),
#             node_indices,
#         ),
#     )


# def make_symmetric_bc(
#     node_indices: jax.Array | Sequence[int],
# ) -> Tuple[BoundaryCondition, ...]:
#     """Create a symmetry (zero-flux Neumann) boundary condition.

#     Sets ``beta * du/dn = 0`` with ``beta = 1``, enforcing zero normal
#     derivative at the specified nodes.

#     Parameters
#     ----------
#     node_indices : jax.Array | Sequence[int]
#         Indices of the grid nodes where the BC is applied.

#     Returns
#     -------
#     tuple of BoundaryCondition
#         A single-element tuple containing the Neumann BC.
#     """
#     return (
#         BoundaryCondition(
#             0.0,
#             1.0,
#             unxt.Quantity(0.0, "V"),
#             node_indices,
#         ),
#     )


# def make_stern_bc(
#     node_indices: jax.Array | Sequence[int],
#     *,
#     phi: unxt.Quantity,
#     eps_gc: float,
#     eps_s: float,
#     d_s: unxt.Quantity,
# ) -> Tuple[BoundaryCondition, ...]:
#     """Create a Stern-layer (Robin) boundary condition.

#     Models the potential drop across a compact Stern layer by setting
#     ``alpha = 1`` and ``beta = -(eps_gc / eps_s) * d_s``, which couples the
#     surface potential to the normal electric field at the outer Helmholtz
#     plane.

#     Parameters
#     ----------
#     node_indices : jax.Array | Sequence[int]
#         Indices of the grid nodes where the BC is applied.
#     phi : unxt.Quantity
#         Potential at the boundary, relative to the potential of zero charge
#         (PZC).
#     eps_gc : float
#         Relative permittivity of the diffuse (Gouy-Chapman) layer.
#     eps_s : float
#         Relative permittivity of the Stern layer.
#     d_s : unxt.Quantity
#         Thickness of the Stern layer.

#     Returns
#     -------
#     tuple of BoundaryCondition
#         A single-element tuple containing the Robin BC.
#     """

#     return (
#         BoundaryCondition(
#             1.0,
#             -(eps_gc / eps_s) * d_s,
#             -phi,
#             node_indices,
#         ),
#     )
