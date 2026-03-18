# SPDX-License-Identifier: GPL-3.0-or-later
class Electrolyte:
    """
    Electrolyte component of an EDL simulation.

    Parameters
    ----------
    global_params : dict
        Global simulation parameters set as instance attributes.
    electroneutrality : bool, optional
        If ``True`` (default), the bulk solution is verified to be
        electrically neutral via :meth:`_electroneutrality_condition`.
    **kwargs
        Additional keyword arguments set as instance attributes (e.g.,
        ``ions`` – a dict mapping ion names to
        ``{"charge": z, "conc": c_bulk}`` entries).

    Attributes
    ----------
    ions : dict
        Mapping of ion names to their charge number and bulk concentration.

    Raises
    ------
    ValueError
        If ``electroneutrality`` is ``True`` and the net charge
        concentration does not sum to zero.
    """

    def __init__(
        self,
        global_params: dict,
        electroneutrality: bool = True,
        **kwargs,
    ):
        # set global params as attributes
        for key, value in global_params.items():
            setattr(self, key, value)
        # set all kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        if electroneutrality:
            self._electroneutrality_condition()

    def _electroneutrality_condition(self) -> None:
        """
        Verify that the bulk solution is electrically neutral.

        Raises
        ------
        ValueError
            If the weighted sum of charge × concentration over all ionic
            species is non-zero.
        """
        # Placeholder for electroneutrality condition logic
        tot_charge = 0.0
        for data in self.ions.values():
            q = data["charge"]
            c = data["conc"]
            # Implement logic to adjust concentrations to satisfy electroneutrality
            tot_charge += q * c
        if tot_charge != 0.0:
            raise ValueError(
                "Electroneutrality condition not satisfied. Total charge concentration: {}".format(
                    tot_charge
                )
            )
