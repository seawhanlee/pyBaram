# -*- coding: utf-8 -*-
from pybaram.solvers.rans import RANSElements
from pybaram.solvers.navierstokes import ViscousFluidElements
from pybaram.utils.nb import dot
from pybaram.utils.np import eps

import functools as fc


class RANSSAFluidElements(ViscousFluidElements):
    name = 'rans-sa'
    nturbvars = 1

    @property
    def auxvars(self):
        return ['ydns', 'mu', 'mut']

    @property
    def primevars(self):
        return super().primevars + ['nut']

    @property
    def conservars(self):
        return super().conservars + ['nut']

    def prim_to_conv(self, pri, cfg):
        return super().prim_to_conv(pri, cfg) + [pri[-1]]

    def conv_to_prim(self, con, cfg):
        return super().conv_to_prim(con, cfg) + [con[-1]]

    @fc.lru_cache()
    def mut_container(self):
        nvars = self.nvars

        # Constants
        cv1 = self._turb_coeffs['cv1']
        cv13 = cv1**3

        def mut(u, g, mu, d):
            # Dynamic viscosity
            rho, nut = u[0], u[nvars-1]
            nu = mu/rho

            # functions
            xi = nut / nu
            fv1 = xi**3 / (xi**3 + cv13)

            return max(rho*nut*fv1, 0)

        return self.be.compile(mut)

    def tflux_container(self):
        ndims, nvars = self.ndims, self.nvars

        def tflux(u, nf, f):
            # Convective flux for turbulent variables
            rho = u[0]
            contrav = dot(u, nf, ndims, 1, 0)/rho

            f[0] = u[nvars-1]*contrav

        return self.be.compile(tflux)

    def turb_src_container(self):
        from pybaram.solvers.rans.turbulent import make_vorticity
        from pybaram.utils.np import eps
        
        ndims, nvars = self.ndims, self.nvars

        # Turbulent constants
        cv1 = self._turb_coeffs['cv1']
        cb1, cb2 = self._turb_coeffs['cb1'], self._turb_coeffs['cb2']
        cw2, cw3 = self._turb_coeffs['cw2'], self._turb_coeffs['cw3']
        sigma, kappa = self._turb_coeffs['sigma'], self._turb_coeffs['kappa']

        cv13 = cv1**3
        cw1 = cb1/kappa**2 + (1 + cb2)/sigma
        
        # Functions
        cplargs = {'ndims' : self.ndims, 'nvars' : self.nvars, 
                    **self._turb_coeffs}
        _vorticity = make_vorticity(self.be, cplargs)

        def src(uc, gc, mu, mut, d, rhs, dsrc):
            # nut and dnut
            nut = uc[nvars-1]
            dnut2 = 0
            for i in range(ndims):
                dnut2 += gc[i][nvars-1]**2

            # Magnitude of vorticity
            omega = _vorticity(uc, gc)

            # SA functions
            nu = mu / uc[0]
            xi = nut / nu
            fv1 = xi**3 / (xi**3 + cv13)
            fv2 = 1 - nut / (nu + nut*fv1)
            Shat = omega + nut/(kappa*d)**2*fv2
            Shat = max(Shat, 1000*eps)
        
             # Production
            prod = cb1*Shat*nut

            # Destruction
            r = min(nut/(Shat*(kappa*d)**2), 10)
            g = r + cw2*(r**6 - r)
            glim = ((1 + cw3**6)/(g**6 + cw3**6))**(1/6)
            fw = g*glim
            dest = cw1*fw*(nut/d)**2

            # Difference
            diff = cb2/sigma*dnut2

            # Implicit term
            ddest = cw1*2*fw*nut/d**2

            rhs[nvars-1] += prod - dest + diff
            dsrc[nvars-1] = max(ddest, 0)  # - dprod
        
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
            
            # Prevent negative turbulent variable
            u[nvars-1] = max(eps, u[nvars-1])
            #if u[nvars-1] < 10*eps:
            #    u[nvars-1] = 10*eps

        return self.be.compile(fix_nonPhy)


class RANSSAElements(RANSElements, RANSSAFluidElements):
    def __init__(self, be, cfg, name, eles):
        super().__init__(be, cfg, name, eles)

        # SA Constants
        # See https://turbmodels.larc.nasa.gov/spalart.html#sa
        sect = 'solver-turbulence-coefficients'
        cfg.get(sect, 'cv1', '7.1')
        cfg.get(sect, 'cb1', '0.1355')
        cfg.get(sect, 'cb2', '0.622')
        cfg.get(sect, 'sigma', '2/3')
        cfg.get(sect, 'kappa', '0.41')
        cfg.get(sect, 'cw2', '0.3')
        cfg.get(sect, 'cw3', '2')
        cfg.get(sect, 'ct3', '1.2')
        cfg.get(sect, 'ct4', '0.5')

        self._const = cfg.items('constants')
        self._turb_coeffs = cfg.items(sect)
    
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
                mut[idx] = _compute_mut(upts[:, idx], None, mu[idx], None)

        return self.be.make_loop(self.neles, post)   

    def make_turb_wave_speed(self):
        # Dimensions and constants
        ndims, nvars = self.ndims, self.nvars
        sigma = float(self._turb_coeffs['sigma'])

        def _lambdaf(u, nf, rcp_dx, mu, mut):
            rho = u[0]
            contra = dot(u, nf, ndims, 1, 0)/rho

            nu = mu/rho
            nut = u[nvars-1]

            # Wave speed : abs(Vn) + 1/dx/sigma*(nu+nut)
            return abs(contra) + rcp_dx*(nu + nut)/sigma

        return self.be.compile(_lambdaf)

    def make_turb_jacobian(self, sign='positive'):
        ndims, nvars = self.ndims, self.nvars
        sigma = float(self._turb_coeffs['sigma'])

        if sign == 'positive':
            op = 1.0
        elif sign == 'negative':
            op = -1.0
        else:
            raise ValueError("Wrong sign of turbulent jacobian")

        # Compute turbulence Jacobian
        # Upwind scheme applied
        def _jacobian(um, nf, A, rcp_dx, mu, mut, gf, ydnsi):
            rho = um[0]
            contra = dot(um, nf, ndims, 1, 0)/rho
            nu = mu/rho
            nut = um[nvars-1]

            contrap = 0.5*(contra + op*abs(contra))
            A[0][0] = contrap + op*rcp_dx*(nu + nut)/sigma

        return self.be.compile(_jacobian)
    
    def make_source_jacobian(self):
        nvars = self.nvars

        def _dsrc(uf, A, dsrc):
            A[0][0] += dsrc[nvars-1]
        
        return self.be.compile(_dsrc)