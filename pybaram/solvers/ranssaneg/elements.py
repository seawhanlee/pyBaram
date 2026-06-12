# -*- coding: utf-8 -*-
from pybaram.solvers.ranssa.elements import RANSSAElements, RANSSAFluidElements
from pybaram.utils.nb import dot
from pybaram.utils.np import eps

import functools as fc


class RANSSANegFluidElements(RANSSAFluidElements):
    name = 'rans-sa-neg'


    @fc.lru_cache()
    def mut_container(self):
        nvars = self.nvars

        cv1 = self._turb_coeffs['cv1']
        cv13 = cv1**3

        def mut(u, g, mu, d):
            rho, nut = u[0], u[nvars-1]
            if nut < 0:
                return 0.0

            nu = mu/rho
            xi = nut / nu
            fv1 = xi**3 / (xi**3 + cv13)

            return max(rho*nut*fv1, 0)

        return self.be.compile(mut)

    def turb_src_container(self):
        from pybaram.solvers.rans.turbulent import make_vorticity

        ndims, nvars = self.ndims, self.nvars

        cv1 = self._turb_coeffs['cv1']
        cb1, cb2 = self._turb_coeffs['cb1'], self._turb_coeffs['cb2']
        cw2, cw3 = self._turb_coeffs['cw2'], self._turb_coeffs['cw3']
        ct3 = self._turb_coeffs['ct3']
        sigma, kappa = self._turb_coeffs['sigma'], self._turb_coeffs['kappa']

        cv13 = cv1**3
        cw1 = cb1/kappa**2 + (1 + cb2)/sigma

        cplargs = {'ndims' : self.ndims, 'nvars' : self.nvars,
                   **self._turb_coeffs}
        _vorticity = make_vorticity(self.be, cplargs)

        def src(uc, gc, mu, mut, d, rhs, dsrc):
            nut = uc[nvars-1]
            dnut2 = 0
            for i in range(ndims):
                dnut2 += gc[i][nvars-1]**2

            omega = _vorticity(uc, gc)
            nu = mu / uc[0]
            xi = nut / nu

            if nut < 0:
                prod = cb1*(1 - ct3)*omega*nut
                dest = -cw1*(nut/d)**2
                diff = cb2/sigma*dnut2

                dsrc[nvars-1] = max(
                    -(cb1*(1 - ct3)*omega + 2*cw1*nut/d**2), 0
                )
                rhs[nvars-1] += prod - dest + diff
                return

            fv1 = xi**3 / (xi**3 + cv13)
            fv2 = 1 - nut / (nu + nut*fv1)
            Sbar = nut/(kappa*d)**2*fv2
            c2, c3 = 0.7, 0.9
            if Sbar >= -c2*omega:
                Shat = omega + Sbar
            else:
                Shat = omega + omega*(c2**2*omega + c3*Sbar)/(
                    (c3 - 2*c2)*omega - Sbar
                )
            Shat = max(Shat, 1e-10)

            prod = cb1*Shat*nut

            r = min(nut/(Shat*(kappa*d)**2), 10)
            g = r + cw2*(r**6 - r)
            glim = ((1 + cw3**6)/(g**6 + cw3**6))**(1/6)
            fw = g*glim
            dest = cw1*fw*(nut/d)**2

            diff = cb2/sigma*dnut2
            ddest = cw1*2*fw*nut/d**2

            rhs[nvars-1] += prod - dest + diff
            dsrc[nvars-1] = max(ddest, 0)

        return self.be.compile(src)

    def fix_nonPys_container(self):
        gamma, pmin = self._const['gamma'], self._const['pmin']
        ndims, nfvars = self.ndims, self.nfvars

        def fix_nonPhy(u):
            rho, et = u[0], u[nfvars-1]
            if rho < 0:
                u[0] = rho = eps

            p = (gamma - 1)*(et - 0.5*dot(u, u, ndims, 1, 1)/rho)

            if p < pmin:
                u[nfvars - 1] = pmin/(gamma-1) + 0.5*dot(u, u, ndims, 1, 1)/rho

        return self.be.compile(fix_nonPhy)


class RANSSANegElements(RANSSAElements, RANSSANegFluidElements):
    name = 'rans-sa-neg'

    def __init__(self, be, cfg, name, eles):
        cfg.get('solver-turbulence-coefficients', 'cn1', '16')
        super().__init__(be, cfg, name, eles)

    def make_turb_wave_speed(self):
        ndims, nvars = self.ndims, self.nvars
        sigma = float(self._turb_coeffs['sigma'])
        cn1 = self._turb_coeffs['cn1']

        def _lambdaf(u, nf, rcp_dx, mu, mut):
            rho = u[0]
            contra = dot(u, nf, ndims, 1, 0)/rho

            nu = mu/rho
            nut = u[nvars-1]
            nusa = nu + nut

            if nut < 0:
                chi3 = (nut/nu)**3
                fneg = (cn1 + chi3)/(cn1 - chi3)
                nusa = nu + nut*fneg

            return abs(contra) + rcp_dx*nusa/sigma

        return self.be.compile(_lambdaf)

    def make_turb_jacobian(self, sign='positive'):
        ndims, nvars = self.ndims, self.nvars
        sigma = float(self._turb_coeffs['sigma'])
        cn1 = self._turb_coeffs['cn1']

        if sign == 'positive':
            op = 1.0
        elif sign == 'negative':
            op = -1.0
        else:
            raise ValueError("Wrong sign of turbulent jacobian")

        def _jacobian(um, nf, A, rcp_dx, mu, mut, gf, ydnsi):
            rho = um[0]
            contra = dot(um, nf, ndims, 1, 0)/rho
            nu = mu/rho
            nut = um[nvars-1]
            nusa = nu + nut

            if nut < 0:
                chi3 = (nut/nu)**3
                fneg = (cn1 + chi3)/(cn1 - chi3)
                nusa = nu + nut*fneg

            contrap = 0.5*(contra + op*abs(contra))
            A[0][0] = contrap + op*rcp_dx*nusa/sigma

        return self.be.compile(_jacobian)
