# -*- coding: utf-8 -*-
from pybaram.solvers.baseadvecdiff import BaseAdvecDiffElements
from pybaram.backends.types import Kernel
from pybaram.solvers.rans.walldistance import compute_wall_distance
from pybaram.utils.nb import dot
from pybaram.utils.np import npeval

import numpy as np


class RANSElements(BaseAdvecDiffElements):
    def __init__(self, be, cfg, name, eles):
        super().__init__(be, cfg, name, eles)
        self.nvars = len(self.primevars)
        self.nfvars = self.nvars - self.nturbvars

        # Constants
        cfg.get('constants', 'pmin', '1e-15')
        self._const = cfg.items('constants')

    def set_ics_from_cfg(self, btri):
        xc = self.geom.xc(self.eles).T

        # Calculate wall distance
        self._ydist = ydist = compute_wall_distance(
            self.be, self.ndims, self.neles, self.xc, btri
        )

        # Parse initial condition from expressions
        subs = dict(zip('xyz', xc))
        subs.update({'ydist': ydist})
        ics = [npeval(self.cfg.getexpr('soln-ics', v, self._const), subs)
               for v in self.primevars]
        ics = self.prim_to_conv(ics, self.cfg)

        # Allocate numpy array and copy parsed values
        self._ics = np.empty((self.nvars, self.neles))
        for i in range(self.nvars):
            self._ics[i] = ics[i]        

    def construct_kernels(self, vertex, nreg, impl_op):
        is_aux_initialized = self._construct_aux_arrays()

        # Call parent method
        super().construct_kernels(vertex, nreg)

        self._construct_impl_arrays(impl_op)
        self._bind_aux_arrays()

        # Update arguments of post kernel
        self.post.update_args(
            self.ydist, self.upts_in, self.grad, self.mu, self.mut
        )

        if not is_aux_initialized:
            # Initialize viscosity
            self.post()

        # Update arguments of divergence kernel
        div_args = (
            *self._div_upts_args, self.upts_out, self.fpts, self.upts_in,
            self.grad, self.dsrc, self.mu, self.mut, self.ydist
        )
        self.div_upts.update_args(*div_args)

        # Kernel to compute timestep
        self.timestep = Kernel(*self._make_timestep(),
                               self.upts_in, self.mu, self.mut, self.dt)

    def _construct_aux_arrays(self):
        # Raw-prefixed arrays are host-side numpy arrays.
        nauxvars = len(self.auxvars)
        self.rawaux = rawaux = np.empty((nauxvars, self.neles))

        # Assign aux variables
        self.rawydist, self.rawmu, self.rawmut = rawaux

        if hasattr(self, "_aux"):
            self.rawaux[:] = self._aux
            delattr(self, "_aux")
            return True

        self.rawaux[0] = self._ydist
        delattr(self, '_ydist')
        return False

    def _construct_impl_arrays(self, impl_op):
        if impl_op == 'spectral-radius':
            # Spectral radius (flow and turbulent model)
            self.fspr = self.be.alloc_array((self.nface, self.neles))
            self.tfspr = self.be.alloc_array((self.nface, self.neles))
        elif impl_op == 'approx-jacobian':
            # Jacobian matrices (flow and turbulent model)
            # 2-dimensional arrays (FVS and Upwind)
            self.jmat = self.be.alloc_array(
                (2, self.nfvars, self.nfvars, self.nface, self.neles)
            )
            self.tjmat = self.be.alloc_array(
                (2, self.nturbvars, self.nturbvars, self.nface, self.neles)
            )

    def _bind_aux_arrays(self):
        self.aux = self.be.convert_array(self.rawaux)
        self.ydist, self.mu, self.mut = self.aux

    def make_wave_speed(self):
        # Dimensions and constants
        ndims, nfvars = self.ndims, self.nfvars
        gamma, pmin = self._const['gamma'], self._const['pmin']
        pr, prt = self._const['pr'], self._const['prt']

        def _lambdaf(u, nf, rcp_dx, mu, mut):
            rho, et = u[0], u[nfvars-1]

            contra = dot(u, nf, ndims, 1, 0)/rho
            p = max((gamma - 1)*(et - 0.5*dot(u, u, ndims, 1, 1)/rho), pmin)
            c = np.sqrt(gamma*p/rho)

            # Wave speed abs(Vn) + c + 1/dx/rho * max(4/3 \gamma) (mu/pr + mut/prt)
            return abs(contra) + c + rcp_dx/rho*max(4/3, gamma)*(mu/pr + mut/prt)

        return self.be.compile(_lambdaf)

    def _make_timestep(self):
        # Dimensions
        ndims, nface = self.ndims, self.nface
        nfvars = self.nfvars

        # Static variables
        vol = self.vol
        _smag, _svec = self._gen_snorm_fpts()
        smag = self.be.convert_array(_smag)
        svec = self.be.convert_array(_svec)

        # Constants
        gamma, pmin = self._const['gamma'], self._const['pmin']
        pr, prt = self._const['pr'], self._const['prt']

        def timestep(i_begin, i_end, smag, svec, vol, u, mu, mut, dt, cfl):
            for idx in range(i_begin, i_end):
                rho = u[0, idx]
                et = u[nfvars-1, idx]
                rv2 = dot(u[:, idx], u[:, idx], ndims, 1, 1)/rho

                p = max((gamma - 1)*(et - 0.5*rv2), pmin)
                c = np.sqrt(gamma*p/rho)

                # Sum of inviscid and viscous spectral radii on faces
                lamc, lamv = 0.0, 0.0
                for jdx in range(nface):
                    # Inviscid spectral radius: Wave speed abs(Vn) + c
                    lamc += (abs(dot(u[:, idx], svec[jdx, idx], ndims, 1, 0))/rho + c)*smag[jdx, idx]

                    # Viscous spectral radius
                    lamv += (1/rho*max(4/3, gamma)*(mu[idx]/pr + mut[idx]/prt)*
                              smag[jdx, idx]**2/vol[idx])

                # Time step : CFL * vol / max(lam_c, C*lam_v), C=4
                dt[idx] = cfl*vol[idx] / max(lamc, 4*lamv)

        return self.be.make_loop(self.neles, timestep, smag, svec, vol)

    def _make_recon(self):
        nface, ndims = self.nface, self.ndims
        nvars, nfvars = self.nvars, self.nfvars
        op = self.be.convert_array(self.dxf)

        def _cal_recon(i_begin, i_end, op, upts, grad, lim, fpts):
            for i in range(i_begin, i_end):
                for l in range(nfvars):
                    for k in range(nface):
                        tmp = 0
                        for j in range(ndims):
                            tmp += op[k, j, i]*grad[j, l, i]
                        fpts[k, l, i] = upts[l, i] + lim[l, i]*tmp

                # First order reconstruction for turbulent variables
                for l in range(nfvars, nvars):
                    for k in range(nface):
                        fpts[k, l, i] = upts[l, i]

        return self.be.make_loop(self.neles, _cal_recon, op)

    def _make_div_upts(self):
        turb_src = self.turb_src_container()
        src, has_xc = self._source_exprs()

        rcp_vol = self.be.convert_array(self.rcp_vol)
        args = 'rcp_vol, rhs, fpts, upts, grad, dsrc, mu, mut, ydist'
        if has_xc:
            xc = self.be.convert_array(self.xc.T)
            self._div_upts_args = (rcp_vol, xc)
            args = 'rcp_vol, xc, rhs, fpts, upts, grad, dsrc, mu, mut, ydist'
        else:
            self._div_upts_args = (rcp_vol,)

        f_txt = (
            f"def _div_upts(i_begin, i_end, {args}, t=0):\n"
            f"    for idx in range(i_begin, i_end):\n"
            f"        rcp_voli = rcp_vol[idx]\n"
        )
        for j, s in enumerate(src):
            subtxt = "+".join("fpts[{},{},idx]".format(i, j)
                              for i in range(self.nface))
            f_txt += "        rhs[{}, idx] = -rcp_voli*({}) + {}\n".format(
                j, subtxt, s)

        f_txt += (
            f"\n"
            f"        # Turbulence source term\n"
            f"        turb_src(upts[:, idx], grad[:, :, idx], mu[idx], mut[idx],\n"
            f"                 ydist[idx], rhs[:, idx], dsrc[:, idx])"
        )

        lvars = {}
        exec(f_txt, {"np": np, "turb_src": turb_src}, lvars)

        # Compile the function
        if has_xc:
            return self.be.make_loop(self.neles, lvars["_div_upts"],
                                     rcp_vol, xc, src=f_txt)
        else:
            return self.be.make_loop(self.neles, lvars["_div_upts"],
                                     rcp_vol, src=f_txt)

    def _make_post(self):
        # Get post-process function
        _fix_nonphys = self.fix_nonPys_container()
        _compute_mu = self.mu_container()
        _compute_mut = self.mut_container()

        muf = self._const['mu']

        def post(i_begin, i_end, ydist, upts, grad, mu, mut):
            # Apply the function over eleemnts
            for idx in range(i_begin, i_end):
                _fix_nonphys(upts[:, idx])
                mu[idx] = _compute_mu(upts[:, idx])
                mut[idx] = _compute_mut(
                    upts[:, idx], grad[:,:,idx], mu[idx], ydist[idx]
                )

                # Limit for turbulence viscosity
                mut[idx] = min(mut[idx], 100000*muf)

        return self.be.make_loop(self.neles, post)
