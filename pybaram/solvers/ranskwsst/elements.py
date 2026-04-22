# -*- coding: utf-8 -*-
from pybaram.solvers.rans import RANSElements
from pybaram.solvers.navierstokes import ViscousFluidElements
from pybaram.utils.nb import dot
from pybaram.utils.np import eps

import functools as fc
import numpy as np


class RANSKWSSTFluidElements(ViscousFluidElements):
    name = 'rans-kwsst'
    nturbvars = 2

    @property
    def auxvars(self):
        return ['ydist', 'mu', 'mut']

    @property
    def primevars(self):
        return super().primevars + ['k', 'omega']

    @property
    def conservars(self):
        return super().conservars + ['rhok', 'rhoomega']

    def prim_to_conv(self, pri, cfg):
        return super().prim_to_conv(pri, cfg) + [pri[-2]*pri[0], pri[-1]*pri[0]]

    def conv_to_prim(self, con, cfg):
        return super().conv_to_prim(con, cfg) + [con[-2]/con[0], con[-1]/con[0]]    

    @fc.lru_cache()
    def mut_container(self):
        from pybaram.solvers.rans.turbulent import make_vorticity
        from pybaram.solvers.ranskwsst.turbulent import make_blendingF2
        
        cplargs = {'ndims' : self.ndims, 'nvars' : self.nvars, 
                    **self._turb_coeffs}

        # Functions
        _vorticity = make_vorticity(self.be, cplargs)
        _f2 = make_blendingF2(self.be, cplargs)

        a1 = self._turb_coeffs['a1']
        mut_max = self._turb_coeffs['mut_limit']*self._const['mu']

        def _mut(uc, gc, mu, d):
            w = uc[-1] / uc[0]
            rk = uc[-2]

            omega = _vorticity(uc, gc)
            f2 = _f2(uc, mu, d)

            # Turbulence viscosity
            mut = a1*rk / max(a1*w, f2*omega)

            # Limit mut (non-zero, below muf*limit)
            return min(max(eps, mut), mut_max)

        return self.be.compile(_mut)

    def tflux_container(self):
        ndims, nvars = self.ndims, self.nvars

        def tflux(u, nf, f):
            # Convective flux for turbulent variables
            rho = u[0]
            contrav = dot(u, nf, ndims, 1, 0)/rho

            f[0] = u[nvars-2]*contrav
            f[1] = u[nvars-1]*contrav

        return self.be.compile(tflux)

    def turb_src_container(self):
        from pybaram.solvers.rans.turbulent import make_vorticity
        from pybaram.solvers.ranskwsst.turbulent import make_blendingF1

        cplargs = {'ndims' : self.ndims, 'nvars' : self.nvars, 
                    **self._turb_coeffs}

        # Functions
        _vorticity = make_vorticity(self.be, cplargs)
        _f1 = make_blendingF1(self.be, cplargs)

        # Constants
        nvars, ndims = self.nvars, self.ndims
        betast = self._turb_coeffs['betast']
        beta1, beta2 = self._turb_coeffs['beta1'], self._turb_coeffs['beta2']
        tgamma1, tgamma2 = self._turb_coeffs['tgamma1'], self._turb_coeffs['tgamma2']
        sigmaw2 = self._turb_coeffs['sigmaw2']
        
        def src(uc, gc, mu, mut, d, rhs, dsrc):
            rho = uc[0]
            k = uc[nvars-2] / rho
            w = uc[nvars-1] / rho
            nut = mut / rho

            # Compute dk/dx_i dw/dx_i
            kwcross = 0
            for i in range(ndims):
                rho_x = gc[i][0]
                k_x = (gc[i][nvars-2] - k*rho_x)/rho
                w_x = (gc[i][nvars-1] - w*rho_x)/rho
                kwcross += k_x*w_x

            # Vorticity
            omega = _vorticity(uc, gc)

            # SST-Vm
            bigP = mut*omega**2

            # Blending function
            f1 = _f1(uc, gc, mu, d)
            tgamma = f1*tgamma1 + (1-f1)*tgamma2
            beta = f1*beta1 + (1-f1)*beta2

            prodk = min(bigP, 20*betast*rho*w*k)
            ddestk = betast*w 
            destk = ddestk*rho*k

            prodw = tgamma / nut * prodk
            crossw = 2*(1-f1)*rho*sigmaw2/w*kwcross 
            ddestw = 2*beta*w + max(crossw, 0)/(rho*w)
            destw = beta*rho*w**2 - crossw

            rhs[nvars-2] += prodk - destk
            rhs[nvars-1] += prodw - destw

            dsrc[nvars-2] = max(ddestk, 0)
            dsrc[nvars-1] = max(ddestw, 0)

        return self.be.compile(src)

    def fix_nonPys_container(self):
        # Constants and dimensions
        gamma, pmin = self._const['gamma'], self._const['pmin']
        ndims, nfvars, nvars = self.ndims, self.nfvars, self.nvars

        def fix_nonPhy(u):
            # Fix non-physical solution (negative density, pressure)
            rho, et = u[0], u[nfvars-1]
            if rho < 0:
                u[0] = rho = eps

            p = (gamma - 1)*(et - 0.5*dot(u, u, ndims, 1, 1)/rho)

            if p < pmin:
                u[nfvars - 1] = pmin/(gamma-1) + 0.5*dot(u, u, ndims, 1, 1)/rho
            
            # Prevent negative turbulent variables
            u[nvars-2] = max(eps, u[nvars-2])
            u[nvars-1] = max(eps, u[nvars-1])

        return self.be.compile(fix_nonPhy)


class RANSKWSSTElements(RANSElements, RANSKWSSTFluidElements):
    def __init__(self, be, cfg, name, eles):
        super().__init__(be, cfg, name, eles)

        # KW-SST Constants
        # See https://turbmodels.larc.nasa.gov/sst.html
        sect = 'solver-turbulence-coefficients'
        cfg.get(sect, 'sigmak1', '0.85')
        cfg.get(sect, 'sigmaw1', '0.5')
        cfg.get(sect, 'beta1', '0.075')
        cfg.get(sect, 'sigmak2', '1.0')
        cfg.get(sect, 'sigmaw2', '0.856')
        cfg.get(sect, 'beta2', '0.0828')
        cfg.get(sect, 'betast', '0.09')
        cfg.get(sect, 'kappa', '0.41')
        cfg.get(sect, 'a1', '0.31')

        # Turbulent viscosity
        cfg.get(sect, 'mut_limit', '1e5')
        
        self._turb_coeffs = cfg.items(sect)

        # Compute gamma1, gamma2
        beta1 = self._turb_coeffs['beta1']
        beta2 = self._turb_coeffs['beta2']
        betast = self._turb_coeffs['betast']
        kappa = self._turb_coeffs['kappa']
        sigmaw1 = self._turb_coeffs['sigmaw1']
        sigmaw2 = self._turb_coeffs['sigmaw2']

        self._turb_coeffs['tgamma1'] = beta1/betast - sigmaw1*kappa**2/np.sqrt(betast)
        self._turb_coeffs['tgamma2'] = beta2/betast - sigmaw2*kappa**2/np.sqrt(betast)

    def _make_post(self):
        # Get post-process function
        _fix_nonPys = self.fix_nonPys_container()
        _compute_mu = self.mu_container()
        _compute_mut = self.mut_container()

        def post(i_begin, i_end, ydist, upts, grad, mu, mut):
            # Apply the function over eleemnts
            for idx in range(i_begin, i_end):
                _fix_nonPys(upts[:, idx])
                mu[idx] = _compute_mu(upts[:, idx])
                mut[idx] = _compute_mut(
                    upts[:, idx], grad[:,:,idx], mu[idx], ydist[idx]
                )

        return self.be.make_loop(self.neles, post)
    
    def make_turb_wave_speed(self):
        # Dimensions and constants
        ndims = self.ndims
        sigma = 1.0

        def _lambdaf(u, nf, rcp_dx, mu, mut):
            rho = u[0]
            contra = dot(u, nf, ndims, 1, 0)/rho

            # Wave speed : abs(Vn) + 1/dx/rho/sigma*(mu+mut)
            return abs(contra) + rcp_dx*(mu + mut)/rho/sigma

        return self.be.compile(_lambdaf)

    def make_turb_jacobian(self, sign='positive'):
        from pybaram.solvers.ranskwsst.turbulent import make_blendingF1

        # Constants
        cplargs = {'ndims': self.ndims, 'nvars': self.nvars, **self._turb_coeffs}
        ndims = self.ndims
        sigmak1 = self._turb_coeffs['sigmak1']
        sigmak2 = self._turb_coeffs['sigmak2']
        sigmaw1 = self._turb_coeffs['sigmaw1']
        sigmaw2 = self._turb_coeffs['sigmaw2']

        # Sign operator
        if sign == 'positive':
            op = 1.0
        elif sign == 'negative':
            op = -1.0
        else:
            raise ValueError("Wrong sign of turbulent jacobian")

        # Functions
        _f1 = make_blendingF1(self.be, cplargs)
        
        # Compute turbulence Jacobian
        # Upwind scheme applied
        def _jacobian(um, nf, A, rcp_dx, mu, mut, gf, ydnsi):
            f1 = _f1(um, gf, mu, ydnsi)
            sigk = f1*sigmak1 + (1-f1)*sigmak2
            sigw = f1*sigmaw1 + (1-f1)*sigmaw2
            
            rho = um[0]
            contra = dot(um, nf, ndims, 1, 0)/rho
            contrap = 0.5*(contra + op*abs(contra))

            A[0][0] = contrap + op*rcp_dx*(mu + sigk*mut)/rho
            A[0][1] = 0.0
            A[1][0] = 0.0
            A[1][1] = contrap + op*rcp_dx*(mu + sigw*mut)/rho

        return self.be.compile(_jacobian)

    def make_source_jacobian(self):
        nvars = self.nvars
        betast = self._turb_coeffs['betast']

        def _dsrc(uf, A, dsrc):
            k = uf[nvars-2]/uf[0]

            A[0][0] += dsrc[nvars-2]
            A[0][1] += max(betast*k, 0)
            # A[1][0] += 0.0
            A[1][1] += dsrc[nvars-1]

        return self.be.compile(_dsrc)