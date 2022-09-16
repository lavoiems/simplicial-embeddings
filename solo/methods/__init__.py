# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from solo.methods.barlow_twins import BarlowTwins
from solo.methods.sdbt import SDBarlowTwins
from solo.methods.base import BaseMethod
from solo.methods.byol import BYOL
from solo.methods.sdbyol import SDBYOL
from solo.methods.partsdbyol import PartSDBYOL
from solo.methods.gstbyol import GSTBYOL
from solo.methods.REINFORCEbyol import REINFORCEBYOL
from solo.methods.sdmocov2plus import SDMoCoV2Plus
from solo.methods.largebyol import LBYOL
from solo.methods.deepclusterv2 import DeepClusterV2
from solo.methods.dino import DINO
from solo.methods.sddino import SDDINO
from solo.methods.linear_control import LinearModel
from solo.methods.linear_fine import LinearModelFine
from solo.methods.linear_masked import LinearModel as LinearMasked
from solo.methods.linear import LinearModel as LinearBase
from solo.methods.mocov2plus import MoCoV2Plus
from solo.methods.nnbyol import NNBYOL
from solo.methods.nnclr import NNCLR
from solo.methods.nnsiam import NNSiam
from solo.methods.ressl import ReSSL
from solo.methods.simclr import SimCLR
from solo.methods.sdsimclr import SDSimCLR
from solo.methods.simsiam import SimSiam
from solo.methods.supcon import SupCon
from solo.methods.swav import SwAV
from solo.methods.sdswav import SDSwAV
from solo.methods.vibcreg import VIbCReg
from solo.methods.vicreg import VICReg
from solo.methods.sdvicreg import SDVICReg
from solo.methods.wmse import WMSE

METHODS = {
    # base classes
    "base": BaseMethod,
    "linear": LinearModel,
    "linear_fine": LinearModelFine,
    "linear_base": LinearBase,
    "linear_masked": LinearMasked,
    # methods
    "barlow_twins": BarlowTwins,
    "sdbarlow": SDBarlowTwins,
    "byol": BYOL,
    "sdbyol": SDBYOL,
    "partsdbyol": PartSDBYOL,
    "gstbyol": GSTBYOL,
    "reinforcebyol": REINFORCEBYOL,
    "sdmoco": SDMoCoV2Plus,
    "lbyol": LBYOL,
    "deepclusterv2": DeepClusterV2,
    "dino": DINO,
    "sddino": SDDINO,
    "mocov2plus": MoCoV2Plus,
    "nnbyol": NNBYOL,
    "nnclr": NNCLR,
    "nnsiam": NNSiam,
    "ressl": ReSSL,
    "simclr": SimCLR,
    "sdsimclr": SDSimCLR,
    "simsiam": SimSiam,
    "supcon": SupCon,
    "swav": SwAV,
    "sdswav": SDSwAV,
    "vibcreg": VIbCReg,
    "vicreg": VICReg,
    "sdvicreg": SDVICReg,
    "wmse": WMSE,
}
__all__ = [
    "BarlowTwins",
    "SDBYOL",
    "BYOL",
    "BaseMethod",
    "DeepClusterV2",
    "DINO",
    "LinearModel",
    "LinearModelFine",
    "MoCoV2Plus",
    "NNBYOL",
    "NNCLR",
    "NNSiam",
    "ReSSL",
    "SimCLR",
    "SimSiam",
    "SupCon",
    "SwAV",
    "VIbCReg",
    "VICReg",
    "WMSE",
]

try:
    from solo.methods import dali  # noqa: F401
except ImportError:
    pass
else:
    __all__.append("dali")
