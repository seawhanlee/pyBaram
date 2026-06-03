# -*- coding: utf-8 -*-
from pybaram.integrators.unsteady import BaseUnsteadyIntegrator
from pybaram.integrators.steady import BaseSteadyIntegrator
from pybaram.integrators.dts import BaseDTSIntegrator
from pybaram.utils.misc import subclass_by_name


def get_integrator(be, cfg, msh, soln, comm):
    mode = cfg.get('solver-time-integrator', 'mode', 'unsteady')
    stepper = cfg.get('solver-time-integrator', 'stepper', 'tvd-rk3')

    if mode == 'unsteady':
        intg = subclass_by_name(BaseUnsteadyIntegrator, stepper)
    elif mode == 'unsteady-dts':
        intg = subclass_by_name(BaseDTSIntegrator, stepper)
    else:
        intg = subclass_by_name(BaseSteadyIntegrator, stepper)

    return intg(be, cfg, msh, soln, comm)
