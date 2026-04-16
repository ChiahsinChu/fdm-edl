# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from typing import Protocol, Sequence, Tuple

import jax
import unxt

from ..utils import constants
from ..utils.bc import BoundaryCondition
from ..utils.unit_conversion import check_data_type


class BCFactory(Protocol):
    """Protocol for boundary condition factory callables.

    Any callable matching this signature can be used as a boundary condition
    factory in the solver setup.
    """

    def __call__(
        self, node_indices: jax.Array | Sequence[int], /, **kwargs
    ) -> Tuple["BoundaryCondition", ...]: ...


def make_constp_bc(
    node_indices: jax.Array | Sequence[int],
    *,
    phi: unxt.Quantity,
) -> Tuple["BoundaryCondition", ...]:
    """Create a constant-potential (Dirichlet) boundary condition.

    Sets ``alpha * u + gamma = 0`` with ``alpha = 1`` and ``gamma = -phi``,
    enforcing ``u = phi`` at the specified nodes.

    Parameters
    ----------
    node_indices : jax.Array | Sequence[int]
        Indices of the grid nodes where the BC is applied.
    phi : unxt.Quantity
        Potential at the boundary, relative to the potential of zero charge
        (PZC).

    Returns
    -------
    tuple of BoundaryCondition
        A single-element tuple containing the Dirichlet BC.
    """
    check_data_type(phi, "electrical potential")

    return (
        BoundaryCondition(
            1.0,
            0.0,
            -phi,
            node_indices,
        ),
    )


def make_constq_bc(
    node_indices: jax.Array | Sequence[int],
    *,
    sigma: unxt.Quantity,
    eps_r: float = 78.5,
) -> Tuple["BoundaryCondition", ...]:
    """Create a constant-charge (Neumann) boundary condition.

    Sets ``beta * du/dn = sigma / (epsilon_0 * epsilon_r)`` with ``beta = 1``,
    enforcing a fixed normal electric field corresponding to the specified
    surface charge density via Gauss's law.

    Parameters
    ----------
    node_indices : jax.Array | Sequence[int]
        Indices of the grid nodes where the BC is applied.
    sigma : unxt.Quantity
        Surface charge density at the boundary.
    eps_r : float, default 78.5
        Relative permittivity of the electrolyte, used to convert surface
        charge density to electric field via Gauss's law.  The default value
        corresponds to bulk water at room temperature; for other solvents or
        conditions, the appropriate value should be used.

    Returns
    -------
    tuple of BoundaryCondition
        A single-element tuple containing the Neumann BC.
    """
    check_data_type(sigma, "surface charge density")
    return (
        BoundaryCondition(
            0.0,
            1.0,
            -sigma / (constants.VACUUM_PERMITTIVITY * eps_r),
            node_indices,
        ),
    )


def make_symmetric_bc(
    node_indices: jax.Array | Sequence[int],
) -> Tuple["BoundaryCondition", ...]:
    """Create a symmetry (zero-flux Neumann) boundary condition.

    Sets ``beta * du/dn = 0`` with ``beta = 1``, enforcing zero normal
    derivative at the specified nodes.

    Parameters
    ----------
    node_indices : jax.Array | Sequence[int]
        Indices of the grid nodes where the BC is applied.

    Returns
    -------
    tuple of BoundaryCondition
        A single-element tuple containing the Neumann BC.
    """
    return (
        BoundaryCondition(
            0.0,
            1.0,
            unxt.Quantity(0.0, "V"),
            node_indices,
        ),
    )


def make_stern_bc(
    node_indices: jax.Array | Sequence[int],
    *,
    phi: unxt.Quantity,
    eps_gc: float,
    eps_s: float,
    d_s: unxt.Quantity,
) -> Tuple["BoundaryCondition", ...]:
    """Create a Stern-layer (Robin) boundary condition.

    Models the potential drop across a compact Stern layer by setting
    ``alpha = 1`` and ``beta = -(eps_gc / eps_s) * d_s``, which couples the
    surface potential to the normal electric field at the outer Helmholtz
    plane.

    Parameters
    ----------
    node_indices : jax.Array | Sequence[int]
        Indices of the grid nodes where the BC is applied.
    phi : unxt.Quantity
        Potential at the boundary, relative to the potential of zero charge
        (PZC).
    eps_gc : float
        Relative permittivity of the diffuse (Gouy-Chapman) layer.
    eps_s : float
        Relative permittivity of the Stern layer.
    d_s : unxt.Quantity
        Thickness of the Stern layer.

    Returns
    -------
    tuple of BoundaryCondition
        A single-element tuple containing the Robin BC.
    """

    return (
        BoundaryCondition(
            1.0,
            -(eps_gc / eps_s) * d_s,
            -phi,
            node_indices,
        ),
    )
