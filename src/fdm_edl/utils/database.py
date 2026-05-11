# SPDX-License-Identifier: GPL-3.0-or-later
import logging
from dataclasses import dataclass

import unxt

from .unit_conversion import check_data_type


@dataclass(frozen=True)
class IonicRadius:
    name: str
    d_ion_water: unxt.Quantity
    R_ion: unxt.Quantity
    R_ion_Pauling: unxt.Quantity | None

    def __post_init__(self):
        check_data_type(self.d_ion_water, "length")
        check_data_type(self.R_ion, "length")
        if self.R_ion_Pauling is not None:
            check_data_type(self.R_ion_Pauling, "length")


logging.warning(
    "Datasets in this module include DOI references. \n"
    "Please cite the appropriate sources for any data you use. \n"
    "Example to get the DOI: IONIC_RADII_DATA['DOI']\n"
)


IONIC_RADII_DATA = {
    "DOI": "10.1021/cr00090a003",
    "Li+": IonicRadius(
        name="Li+",
        d_ion_water=unxt.Quantity(0.208, "nm"),
        R_ion=unxt.Quantity(0.071, "nm"),
        R_ion_Pauling=unxt.Quantity(0.074, "nm"),
    ),
    "Na+": IonicRadius(
        name="Na+",
        d_ion_water=unxt.Quantity(0.2356, "nm"),
        R_ion=unxt.Quantity(0.097, "nm"),
        R_ion_Pauling=unxt.Quantity(0.102, "nm"),
    ),
    "K+": IonicRadius(
        name="K+",
        d_ion_water=unxt.Quantity(0.2798, "nm"),
        R_ion=unxt.Quantity(0.141, "nm"),
        R_ion_Pauling=unxt.Quantity(0.138, "nm"),
    ),
    "Rb+": IonicRadius(
        name="Rb+",
        d_ion_water=unxt.Quantity(0.289, "nm"),
        R_ion=unxt.Quantity(0.150, "nm"),
        R_ion_Pauling=unxt.Quantity(0.149, "nm"),
    ),
    "Cs+": IonicRadius(
        name="Cs+",
        d_ion_water=unxt.Quantity(0.3139, "nm"),
        R_ion=unxt.Quantity(0.173, "nm"),
        R_ion_Pauling=unxt.Quantity(0.170, "nm"),
    ),
    "Ag+": IonicRadius(
        name="Ag+",
        d_ion_water=unxt.Quantity(0.2417, "nm"),
        R_ion=unxt.Quantity(0.102, "nm"),
        R_ion_Pauling=unxt.Quantity(0.115, "nm"),
    ),
    "H3O+": IonicRadius(
        name="H3O+",
        d_ion_water=unxt.Quantity(0.2755, "nm"),
        R_ion=unxt.Quantity(0.141, "nm"),
        R_ion_Pauling=None,
    ),
    "Mg2+": IonicRadius(
        name="Mg2+",
        d_ion_water=unxt.Quantity(0.2090, "nm"),
        R_ion=unxt.Quantity(0.070, "nm"),
        R_ion_Pauling=unxt.Quantity(0.072, "nm"),
    ),
    "Ca2+": IonicRadius(
        name="Ca2+",
        d_ion_water=unxt.Quantity(0.2422, "nm"),
        R_ion=unxt.Quantity(0.103, "nm"),
        R_ion_Pauling=unxt.Quantity(0.100, "nm"),
    ),
    "Sr2+": IonicRadius(
        name="Sr2+",
        d_ion_water=unxt.Quantity(0.264, "nm"),
        R_ion=unxt.Quantity(0.125, "nm"),
        R_ion_Pauling=unxt.Quantity(0.125, "nm"),
    ),
    "Mn2+": IonicRadius(
        name="Mn2+",
        d_ion_water=unxt.Quantity(0.2192, "nm"),
        R_ion=unxt.Quantity(0.080, "nm"),
        R_ion_Pauling=unxt.Quantity(0.083, "nm"),
    ),
    "Fe2+": IonicRadius(
        name="Fe2+",
        d_ion_water=unxt.Quantity(0.2114, "nm"),
        R_ion=unxt.Quantity(0.072, "nm"),
        R_ion_Pauling=unxt.Quantity(0.078, "nm"),
    ),
    "Co2+": IonicRadius(
        name="Co2+",
        d_ion_water=unxt.Quantity(0.2106, "nm"),
        R_ion=unxt.Quantity(0.072, "nm"),
        R_ion_Pauling=unxt.Quantity(0.075, "nm"),
    ),
    "Ni2+": IonicRadius(
        name="Ni2+",
        d_ion_water=unxt.Quantity(0.2061, "nm"),
        R_ion=unxt.Quantity(0.067, "nm"),
        R_ion_Pauling=unxt.Quantity(0.069, "nm"),
    ),
    "Cu2+": IonicRadius(
        name="Cu2+",
        d_ion_water=unxt.Quantity(0.211, "nm"),
        R_ion=unxt.Quantity(0.072, "nm"),
        R_ion_Pauling=unxt.Quantity(0.073, "nm"),
    ),
    "Zn2+": IonicRadius(
        name="Zn2+",
        d_ion_water=unxt.Quantity(0.2098, "nm"),
        R_ion=unxt.Quantity(0.070, "nm"),
        R_ion_Pauling=unxt.Quantity(0.075, "nm"),
    ),
    "Cd2+": IonicRadius(
        name="Cd2+",
        d_ion_water=unxt.Quantity(0.2301, "nm"),
        R_ion=unxt.Quantity(0.091, "nm"),
        R_ion_Pauling=unxt.Quantity(0.095, "nm"),
    ),
    "Hg2+": IonicRadius(
        name="Hg2+",
        d_ion_water=unxt.Quantity(0.242, "nm"),
        R_ion=unxt.Quantity(0.103, "nm"),
        R_ion_Pauling=unxt.Quantity(0.102, "nm"),
    ),
    "Sn2+": IonicRadius(
        name="Sn2+",
        d_ion_water=unxt.Quantity(0.233, "nm"),
        R_ion=unxt.Quantity(0.094, "nm"),
        R_ion_Pauling=unxt.Quantity(0.093, "nm"),
    ),
    "Al3+": IonicRadius(
        name="Al3+",
        d_ion_water=unxt.Quantity(0.1887, "nm"),
        R_ion=unxt.Quantity(0.050, "nm"),
        R_ion_Pauling=unxt.Quantity(0.053, "nm"),
    ),
    "Y3+": IonicRadius(
        name="Y3+",
        d_ion_water=unxt.Quantity(0.2365, "nm"),
        R_ion=unxt.Quantity(0.097, "nm"),
        R_ion_Pauling=unxt.Quantity(0.101, "nm"),
    ),
    "La3+": IonicRadius(
        name="La3+",
        d_ion_water=unxt.Quantity(0.2528, "nm"),
        R_ion=unxt.Quantity(0.114, "nm"),
        R_ion_Pauling=unxt.Quantity(0.118, "nm"),
    ),
    "Ce3+": IonicRadius(
        name="Ce3+",
        d_ion_water=unxt.Quantity(0.255, "nm"),
        R_ion=unxt.Quantity(0.116, "nm"),
        R_ion_Pauling=unxt.Quantity(0.114, "nm"),
    ),
    "Pr3+": IonicRadius(
        name="Pr3+",
        d_ion_water=unxt.Quantity(0.254, "nm"),
        R_ion=unxt.Quantity(0.115, "nm"),
        R_ion_Pauling=unxt.Quantity(0.114, "nm"),
    ),
    "Nd3+": IonicRadius(
        name="Nd3+",
        d_ion_water=unxt.Quantity(0.2472, "nm"),
        R_ion=unxt.Quantity(0.108, "nm"),
        R_ion_Pauling=unxt.Quantity(0.112, "nm"),
    ),
    "Sm3+": IonicRadius(
        name="Sm3+",
        d_ion_water=unxt.Quantity(0.2448, "nm"),
        R_ion=unxt.Quantity(0.106, "nm"),
        R_ion_Pauling=unxt.Quantity(0.109, "nm"),
    ),
    "Eu3+": IonicRadius(
        name="Eu3+",
        d_ion_water=unxt.Quantity(0.245, "nm"),
        R_ion=unxt.Quantity(0.106, "nm"),
        R_ion_Pauling=unxt.Quantity(0.107, "nm"),
    ),
    "Gd3+": IonicRadius(
        name="Gd3+",
        d_ion_water=unxt.Quantity(0.239, "nm"),
        R_ion=unxt.Quantity(0.100, "nm"),
        R_ion_Pauling=unxt.Quantity(0.106, "nm"),
    ),
    "Tb3+": IonicRadius(
        name="Tb3+",
        d_ion_water=unxt.Quantity(0.2403, "nm"),
        R_ion=unxt.Quantity(0.101, "nm"),
        R_ion_Pauling=unxt.Quantity(0.104, "nm"),
    ),
    "Dy3+": IonicRadius(
        name="Dy3+",
        d_ion_water=unxt.Quantity(0.2370, "nm"),
        R_ion=unxt.Quantity(0.098, "nm"),
        R_ion_Pauling=unxt.Quantity(0.103, "nm"),
    ),
    "Er3+": IonicRadius(
        name="Er3+",
        d_ion_water=unxt.Quantity(0.2363, "nm"),
        R_ion=unxt.Quantity(0.097, "nm"),
        R_ion_Pauling=unxt.Quantity(0.100, "nm"),
    ),
    "Tm3+": IonicRadius(
        name="Tm3+",
        d_ion_water=unxt.Quantity(0.236, "nm"),
        R_ion=unxt.Quantity(0.097, "nm"),
        R_ion_Pauling=unxt.Quantity(0.099, "nm"),
    ),
    "Lu3+": IonicRadius(
        name="Lu3+",
        d_ion_water=unxt.Quantity(0.234, "nm"),
        R_ion=unxt.Quantity(0.095, "nm"),
        R_ion_Pauling=unxt.Quantity(0.097, "nm"),
    ),
    "Cr3+": IonicRadius(
        name="Cr3+",
        d_ion_water=unxt.Quantity(0.1969, "nm"),
        R_ion=unxt.Quantity(0.058, "nm"),
        R_ion_Pauling=unxt.Quantity(0.062, "nm"),
    ),
    "Fe3+": IonicRadius(
        name="Fe3+",
        d_ion_water=unxt.Quantity(0.2031, "nm"),
        R_ion=unxt.Quantity(0.064, "nm"),
        R_ion_Pauling=unxt.Quantity(0.065, "nm"),
    ),
    "Rh3+": IonicRadius(
        name="Rh3+",
        d_ion_water=unxt.Quantity(0.204, "nm"),
        R_ion=unxt.Quantity(0.065, "nm"),
        R_ion_Pauling=None,
    ),
    "In3+": IonicRadius(
        name="In3+",
        d_ion_water=unxt.Quantity(0.2156, "nm"),
        R_ion=unxt.Quantity(0.076, "nm"),
        R_ion_Pauling=unxt.Quantity(0.079, "nm"),
    ),
    "Tl3+": IonicRadius(
        name="Tl3+",
        d_ion_water=unxt.Quantity(0.2231, "nm"),
        R_ion=unxt.Quantity(0.084, "nm"),
        R_ion_Pauling=unxt.Quantity(0.088, "nm"),
    ),
    "Th4+": IonicRadius(
        name="Th4+",
        d_ion_water=unxt.Quantity(0.253, "nm"),
        R_ion=unxt.Quantity(0.114, "nm"),
        R_ion_Pauling=unxt.Quantity(0.106, "nm"),
    ),
    "F-": IonicRadius(
        name="F-",
        d_ion_water=unxt.Quantity(0.2630, "nm"),
        R_ion=unxt.Quantity(0.124, "nm"),
        R_ion_Pauling=unxt.Quantity(0.133, "nm"),
    ),
    "Cl-": IonicRadius(
        name="Cl-",
        d_ion_water=unxt.Quantity(0.3187, "nm"),
        R_ion=unxt.Quantity(0.180, "nm"),
        R_ion_Pauling=unxt.Quantity(0.181, "nm"),
    ),
    "Br-": IonicRadius(
        name="Br-",
        d_ion_water=unxt.Quantity(0.3373, "nm"),
        R_ion=unxt.Quantity(0.198, "nm"),
        R_ion_Pauling=unxt.Quantity(0.196, "nm"),
    ),
    "I-": IonicRadius(
        name="I-",
        d_ion_water=unxt.Quantity(0.3647, "nm"),
        R_ion=unxt.Quantity(0.225, "nm"),
        R_ion_Pauling=unxt.Quantity(0.220, "nm"),
    ),
    "NO3-": IonicRadius(
        name="NO3-",
        d_ion_water=unxt.Quantity(0.316, "nm"),
        R_ion=unxt.Quantity(0.177, "nm"),
        R_ion_Pauling=unxt.Quantity(0.179, "nm"),
    ),
    "ClO4-": IonicRadius(
        name="ClO4-",
        d_ion_water=unxt.Quantity(0.370, "nm"),
        R_ion=unxt.Quantity(0.241, "nm"),
        R_ion_Pauling=unxt.Quantity(0.240, "nm"),
    ),
    "H2PO4-": IonicRadius(
        name="H2PO4-",
        d_ion_water=unxt.Quantity(0.377, "nm"),
        R_ion=unxt.Quantity(0.238, "nm"),
        R_ion_Pauling=unxt.Quantity(0.238, "nm"),
    ),
    "SO42-": IonicRadius(
        name="SO42-",
        d_ion_water=unxt.Quantity(0.3815, "nm"),
        R_ion=unxt.Quantity(0.242, "nm"),
        R_ion_Pauling=unxt.Quantity(0.230, "nm"),
    ),
    "SeO42-": IonicRadius(
        name="SeO42-",
        d_ion_water=unxt.Quantity(0.395, "nm"),
        R_ion=unxt.Quantity(0.256, "nm"),
        R_ion_Pauling=unxt.Quantity(0.243, "nm"),
    ),
    "Mo(W)O42-": IonicRadius(
        name="Mo(W)O42-",
        d_ion_water=unxt.Quantity(0.406, "nm"),
        R_ion=unxt.Quantity(0.267, "nm"),
        R_ion_Pauling=unxt.Quantity(0.254, "nm"),
    ),
}
