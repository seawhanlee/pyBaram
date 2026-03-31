# -*- coding: utf-8 -*-
from pybaram.solvers.base.inters import BaseInters
from pybaram.solvers.rans.inters import RANSIntInters, RANSBCInters, RANSMPIInters
from pybaram.solvers.rans.inters import (RANSSlipWallBCInters, RANSAdiaWallBCInters, RANSIsothermWallBCInters,
                                         RANSSupOutBCInters, RANSSupInBCInters, RANSFarBCInters,
                                         RANSSubOutPBCInters, RANSSubInvBCInters, RANSSubInpttBCInters)
from pybaram.solvers.ranskwsst.bcs import get_bc
from pybaram.utils.nb import dot


class RANSKWSSTInters(BaseInters):
    def __init__(self, be, cfg, elemap, *args, **kwargs):
        super().__init__(be, cfg, elemap, *args, **kwargs)

        self._turb_coeffs = self.ele0._turb_coeffs
        self.nturbvars = self.ele0.nturbvars

    def _make_turb_flux(self):
        from pybaram.solvers.ranskwsst.turbulent import make_blendingF1

        ndims, nvars = self.ndims, self.nvars
        sigmak1, sigmak2 = self._turb_coeffs['sigmak1'], self._turb_coeffs['sigmak2']
        sigmaw1, sigmaw2 = self._turb_coeffs['sigmaw1'], self._turb_coeffs['sigmaw2']

        cplargs = {'ndims' : ndims, 'nvars' : nvars, **self._turb_coeffs}
        _f1 = make_blendingF1(self.be, cplargs)

        def tflux(ul, ur, um, gf, nf, ydist, mu, mut, fn):
             # Convective flux
            contral = dot(ul, nf, ndims, 1, 0)/ul[0]
            contrar = dot(ur, nf, ndims, 1, 0)/ur[0]
            contram = 0.5*(contral + contrar)

            contrap = 0.5*(contram + abs(contram))
            contram = 0.5*(contram - abs(contram))

            # Upwind
            fn[nvars-2] = contrap*ul[nvars-2] + contram*ur[nvars-2]
            fn[nvars-1] = contrap*ul[nvars-1] + contram*ur[nvars-1]

            # Viscous
            f1 = _f1(um, gf, mu, ydist)
            sigmak = f1*sigmak1 + (1-f1)*sigmak2
            sigmaw = f1*sigmaw1 + (1-f1)*sigmaw2

            tauk, tauw = 0, 0
            rho = um[0]
            for i in range(ndims):
                rho_x = gf[i][0]
                k_x = (gf[i][nvars-2] - um[nvars-2]*rho_x/rho)/rho
                w_x = (gf[i][nvars-1] - um[nvars-1]*rho_x/rho)/rho

                tauk += k_x*nf[i]
                tauw += w_x*nf[i]

            fn[nvars-2] -= (mu + sigmak*mut)*tauk
            fn[nvars-1] -= (mu + sigmaw*mut)*tauw
        
        return self.be.compile(tflux)


class RANSKWSSTIntInters(RANSIntInters, RANSKWSSTInters):
    pass


class RANSKWSSTMPIInters(RANSMPIInters, RANSKWSSTInters):
    pass


class RANSKWSSTBCInters(RANSBCInters, RANSKWSSTInters):
    _get_bc = get_bc


class RANSKWSSTSlipWallBCInters(RANSKWSSTBCInters, RANSSlipWallBCInters):
    pass


class RANSKWSSTAdiaWallBCInters(RANSKWSSTBCInters, RANSAdiaWallBCInters):
    pass


class RANSKWSSTIsothermWallBCInters(RANSKWSSTBCInters, RANSIsothermWallBCInters):
    pass


class RANSKWSSTSupOutBCInters(RANSKWSSTBCInters, RANSSupOutBCInters):
    pass


class RANSKWSSTSupInBCInters(RANSKWSSTBCInters, RANSSupInBCInters):
    pass


class RANSKWSSTFarBCInters(RANSKWSSTBCInters, RANSFarBCInters):
    pass


class RANSKWSSTSubOutPBCInters(RANSKWSSTBCInters, RANSSubOutPBCInters):
    pass


class RANSKWSSTSubInvBCInters(RANSKWSSTBCInters, RANSSubInvBCInters):
    pass


class RANSKWSSTSubInpttBCInters(RANSKWSSTBCInters, RANSSubInpttBCInters):
    pass