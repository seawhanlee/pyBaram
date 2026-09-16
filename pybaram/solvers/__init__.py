# -*- coding: utf-8 -*-
from pybaram.solvers.base.system import BaseSystem
from pybaram.solvers.euler.system import EulerSystem
from pybaram.solvers.euler.elements import FluidElements
from pybaram.solvers.navierstokes.system import NavierStokeSystem
from pybaram.solvers.ranssa.system import RANSSASystem
from pybaram.solvers.ranssaneg.system import RANSSANegSystem
from pybaram.solvers.ranskwsst.system import (
    RANSKWSSTSystem, RANSKWSST2003mSystem, RANSKWSSTV2003mSystem
)
from pybaram.utils.misc import subclass_by_name


def get_system(
    be, cfg, msh, soln, comm, nreg, impl_op, rank_layout_req=None,
    rank_layout_name=None
):
    name = cfg.get('solver', 'system')
    return subclass_by_name(BaseSystem, name)(
        be, cfg, msh, soln, comm, nreg, impl_op,
        rank_layout_req, rank_layout_name
    )


def get_fluid(name):
    if name in ['euler']:
        return FluidElements()
    else:
        return subclass_by_name(FluidElements, name)()        
