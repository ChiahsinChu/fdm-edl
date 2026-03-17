# SPDX-License-Identifier: GPL-3.0-or-later
class Electrolyte:
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
