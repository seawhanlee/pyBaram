# -*- coding: utf-8 -*-
from pybaram.solvers.rans.system import RANSSystem
from pybaram.solvers.ranskwsst import (
    RANSKWSSTElements, RANSKWSST2003mElements, RANSKWSSTV2003mElements
)
from pybaram.solvers.ranskwsst import RANSKWSSTIntInters, RANSKWSSTBCInters, RANSKWSSTMPIInters


class RANSKWSSTSystem(RANSSystem):
    name = 'rans-kwsst'
    _elements_cls = RANSKWSSTElements
    _intinters_cls = RANSKWSSTIntInters
    _bcinters_cls = RANSKWSSTBCInters
    _mpiinters_cls = RANSKWSSTMPIInters


class RANSKWSST2003mSystem(RANSKWSSTSystem):
    name = 'rans-kwsst-2003m'
    _elements_cls = RANSKWSST2003mElements


class RANSKWSSTV2003mSystem(RANSKWSSTSystem):
    name = 'rans-kwsst-v2003m'
    _elements_cls = RANSKWSSTV2003mElements
