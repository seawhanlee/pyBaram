# -*- coding: utf-8 -*-
from pybaram.solvers.base.inters import BaseInters
from pybaram.solvers.rans.inters import RANSIntInters, RANSBCInters, RANSMPIInters
from pybaram.solvers.rans.inters import (RANSSlipWallBCInters, RANSAdiaWallBCInters, RANSIsothermWallBCInters,
                                         RANSSupOutBCInters, RANSSupInBCInters, RANSFarBCInters,
                                         RANSSubOutPBCInters, RANSSubInvBCInters, RANSSubInpttBCInters)
from pybaram.solvers.ranssa.bcs import get_bc
from pybaram.utils.nb import dot


class RANSSAInters(BaseInters):
    def __init__(self, be, cfg, elemap, *args, **kwargs):
        super().__init__(be, cfg, elemap, *args, **kwargs)

        self._turb_coeffs = self.ele0._turb_coeffs
        self.nturbvars = self.ele0.nturbvars

    def _make_turb_flux(self):
        ndims, nvars = self.ndims, self.nvars

        sigma = self._turb_coeffs['sigma']

        def tflux(ul, ur, um, gf, nf, ydist, mu, mut, fn):
            # Convective flux
            contral = dot(ul, nf, ndims, 1, 0)/ul[0]
            contrar = dot(ur, nf, ndims, 1, 0)/ur[0]
            contram = 0.5*(contral + contrar)

            contrap = 0.5*(contram + abs(contram))
            contram = 0.5*(contram - abs(contram))

            # Upwind
            fn[nvars-1] = contrap*ul[nvars-1] + contram*ur[nvars-1]

            nu = 2*mu / (ul[0] + ur[0])
            nut = 0.5*(ul[nvars-1] + ur[nvars-1])

            tau = dot(gf[:, nvars-1], nf, ndims, 0, 0)

            fn[nvars-1] -= 1/sigma*(nu + nut)*tau

        return self.be.compile(tflux)


class RANSSAIntInters(RANSIntInters, RANSSAInters):
    pass


class RANSSAMPIInters(RANSMPIInters, RANSSAInters):
    pass


class RANSSABCInters(RANSBCInters, RANSSAInters):
    _get_bc = get_bc


class RANSSASlipWallBCInters(RANSSABCInters, RANSSlipWallBCInters):
    pass


class RANSSAAdiaWallBCInters(RANSSABCInters, RANSAdiaWallBCInters):
    pass


class RANSSAIsothermWallBCInters(RANSSABCInters, RANSIsothermWallBCInters):
    pass


class RANSSASupOutBCInters(RANSSABCInters, RANSSupOutBCInters):
    pass


class RANSSASupInBCInters(RANSSABCInters, RANSSupInBCInters):
    pass


class RANSSAFarBCInters(RANSSABCInters, RANSFarBCInters):
    pass


class RANSSASubOutPBCInters(RANSSABCInters, RANSSubOutPBCInters):
    pass


class RANSSASubInvBCInters(RANSSABCInters, RANSSubInvBCInters):
    pass


class RANSSASubInpttBCInters(RANSSABCInters, RANSSubInpttBCInters):
    pass