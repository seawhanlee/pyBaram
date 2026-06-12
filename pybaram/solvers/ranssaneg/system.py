# -*- coding: utf-8 -*-
from pybaram.solvers.rans.system import RANSSystem
from pybaram.solvers.ranssaneg import RANSSANegElements, RANSSANegIntInters, RANSSANegMPIInters, RANSSANegBCInters


class RANSSANegSystem(RANSSystem):
    name = 'rans-sa-neg'
    _elements_cls = RANSSANegElements
    _intinters_cls = RANSSANegIntInters
    _bcinters_cls = RANSSANegBCInters
    _mpiinters_cls = RANSSANegMPIInters
