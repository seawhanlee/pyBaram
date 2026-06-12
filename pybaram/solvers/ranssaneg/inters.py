# -*- coding: utf-8 -*-
from pybaram.solvers.rans.inters import RANSIntInters, RANSBCInters, RANSMPIInters
from pybaram.solvers.rans.inters import (RANSSlipWallBCInters, RANSAdiaWallBCInters, RANSIsothermWallBCInters,
                                         RANSSupOutBCInters, RANSSupInBCInters, RANSFarBCInters,
                                         RANSSubOutPBCInters, RANSSubInvBCInters, RANSSubInpttBCInters)
from pybaram.solvers.ranssa.bcs import get_bc
from pybaram.solvers.ranssa.inters import RANSSAInters
from pybaram.utils.nb import dot


class RANSSANegInters(RANSSAInters):
    def _make_turb_flux(self):
        ndims, nvars = self.ndims, self.nvars

        sigma = self._turb_coeffs['sigma']
        cn1 = self._turb_coeffs['cn1']

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
            nusa = nu + nut

            if nut < 0:
                chi3 = (nut/nu)**3
                fneg = (cn1 + chi3)/(cn1 - chi3)
                nusa = nu + nut*fneg

            tau = dot(gf[:, nvars-1], nf, ndims, 0, 0)

            fn[nvars-1] -= nusa/sigma*tau

        return self.be.compile(tflux)


class RANSSANegIntInters(RANSIntInters, RANSSANegInters):
    pass


class RANSSANegMPIInters(RANSMPIInters, RANSSANegInters):
    pass


class RANSSANegBCInters(RANSBCInters, RANSSANegInters):
    _get_bc = get_bc


class RANSSANegSlipWallBCInters(RANSSANegBCInters, RANSSlipWallBCInters):
    pass


class RANSSANegAdiaWallBCInters(RANSSANegBCInters, RANSAdiaWallBCInters):
    pass


class RANSSANegIsothermWallBCInters(RANSSANegBCInters, RANSIsothermWallBCInters):
    pass


class RANSSANegSupOutBCInters(RANSSANegBCInters, RANSSupOutBCInters):
    pass


class RANSSANegSupInBCInters(RANSSANegBCInters, RANSSupInBCInters):
    pass


class RANSSANegFarBCInters(RANSSANegBCInters, RANSFarBCInters):
    pass


class RANSSANegSubOutPBCInters(RANSSANegBCInters, RANSSubOutPBCInters):
    pass


class RANSSANegSubInvBCInters(RANSSANegBCInters, RANSSubInvBCInters):
    pass


class RANSSANegSubInpttBCInters(RANSSANegBCInters, RANSSubInpttBCInters):
    pass
