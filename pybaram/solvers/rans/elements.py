# -*- coding: utf-8 -*-
from pybaram.solvers.baseadvecdiff import BaseAdvecDiffElements
from pybaram.backends.types import Kernel
from pybaram.utils.nb import dot

import numpy as np


class RANSElements(BaseAdvecDiffElements):
    def __init__(self, be, cfg, name, eles):
        super().__init__(be, cfg, name, eles)
        self.nvars = len(self.primevars)
        self.nfvars = self.nvars - self.nturbvars

        # Constants
        cfg.get('constants', 'pmin', '1e-15')
        self._const = cfg.items('constants')

    def construct_kernels(self, vertex, xw, nreg, impl_op):
        # Aux array
        nauxvars = len(self.auxvars)
        self.aux = aux = np.empty((nauxvars, self.neles))

        # Assign aux variable
        self.ydist, self.mu, self.mut = aux

        if hasattr(self, "_aux"):
            self.aux[:] = self._aux
            delattr(self, "_aux")
            is_aux_initialized = True
        else:
            is_aux_initialized = False

        # Call paraent method
        super().construct_kernels(vertex, nreg)

        if impl_op == 'spectral-radius':
            # Spectral radius (flow and turbulent model)
            self.fspr = np.empty((self.nface, self.neles))
            self.tfspr = np.empty_like(self.fspr)
        elif impl_op == 'approx-jacobian':
            # Jacobian matrices (flow and turbulent model)
            # 2-dimensional arrays (FVS and Upwind)
            self.jmat = np.empty((2, self.nfvars, self.nfvars, \
                                  self.nface, self.neles))
            self.tjmat = np.empty((2, self.nturbvars, self.nturbvars, \
                                   self.nface, self.neles))

        # Update arguments of post kerenl
        self.post.update_args(self.upts_in, self.grad, self.mu, self.mut)
        
        if not is_aux_initialized:
            # Compute wall distance
            self._wall_distance(xw, self.ydist)
            
            # Initialize viscosity
            self.post()

        # Update arguments of divergence kernel
        self.div_upts.update_args(
            self.upts_out, self.fpts, self.upts_in, self.grad,
            self.dsrc, self.mu, self.mut
        )

        # Kernel to compute timestep
        self.timestep = Kernel(self._make_timestep(),
                               self.upts_in, self.mu, self.mut, self.dt)

    def _wall_distance(self, xw, wdist):
        # Compute wall distance
        try:
            # KDtree version 
            try:
                # pykdtree
                self._wall_distance_kdtree_pykdtree(xw, wdist)
            except:
                # Scipy
                self._wall_distance_kdtree_scipy(xw, wdist)
        except:
            # Brute-force version
            self._wall_distance_bf(xw, wdist)

    def _wall_distance_bf(self, xw, wdist):
        # Dimensions and constants
        nf, ne, nd = self.eles.shape
        nw = xw.shape[0]
        eles = self.eles
        rcp_nf = 1.0 / nf

        # Guess maximum distance
        xmax = 2*(eles.max() - eles.min())

        def _cal_wdist(i_begin, i_end, wdist):
            # Brute-force searching
            for idx in range(i_begin, i_end):
                wd_ele = 0
                for jdx in range(nf):
                    # for all node points
                    xc = eles[jdx, idx]
                    
                    # Compute minimum wall distance for each node
                    wd_node = xmax
                    for kdx in range(nw):
                        xwi = xw[kdx]                      
                        
                        # Compute distance
                        dx = 0
                        for i in range(nd):
                            dx += (xwi[i] - xc[i])**2

                        dx = np.sqrt(dx)
                        wd_node = min(dx, wd_node)

                    # Averaging for cell
                    wd_ele += wd_node

                wd_ele *= rcp_nf
                wdist[idx] = wd_ele

        self.be.make_loop(ne, _cal_wdist)(wdist)
    
    def _wall_distance_kdtree_scipy(self, xw, wdist):
        from scipy.spatial import KDTree
        
        # Build Tree data
        tree = KDTree(xw)

        # Check multi-thread or not
        if self.be.multithread == 'single':
            workers = 1
        else:
            workers = -1

        # Compute wall distance from KDtree
        wdist[:] = np.average(tree.query(self.eles, workers=workers)[0], axis=0)
        
        # Delete tree
        del(tree)

    def _wall_distance_kdtree_pykdtree(self, xw, wdist):
        from pykdtree.kdtree import KDTree

        # Build Tree data
        tree = KDTree(xw)

        # Compute wall distance from KDtree
        d, i = tree.query(self.eles.reshape(-1, self.ndims))
        wdist[:] = np.average(d.reshape(-1, self.neles), axis=0)

        # Delete tree
        del(tree)

    def make_wave_speed(self):
        # Dimensions and constants
        ndims, nfvars = self.ndims, self.nfvars
        gamma, pmin = self._const['gamma'], self._const['pmin']
        pr, prt = self._const['pr'], self._const['prt']

        def _lambdaf(u, nf, rcp_dx, mu, mut):
            rho, et = u[0], u[nfvars-1]

            contra = dot(u, nf, ndims, 1)/rho
            p = max((gamma - 1)*(et - 0.5*dot(u, u, ndims, 1, 1)/rho), pmin)
            c = np.sqrt(gamma*p/rho)

            # Wave speed abs(Vn) + c + 1/dx/rho * max(4/3 \gamma) (mu/pr + mut/prt)
            return abs(contra) + c + rcp_dx/rho*max(4/3, gamma)*(mu/pr + mut/prt)

        return self.be.compile(_lambdaf)

    def _make_timestep(self):
        # Dimensions
        ndims, nface = self.ndims, self.nface
        nflvars = self.nfvars

        # Static variables
        vol = self._vol
        smag, svec = self._gen_snorm_fpts()

        # Constants
        gamma, pmin = self._const['gamma'], self._const['pmin']
        pr, prt = self._const['pr'], self._const['prt']

        def timestep(i_begin, i_end, u, mu, mut, dt, cfl):
            for idx in range(i_begin, i_end):
                rho = u[0, idx]
                et = u[nflvars-1, idx]
                rv2 = dot(u[:, idx], u[:, idx], ndims, 1, 1)/rho

                p = max((gamma - 1)*(et - 0.5*rv2), pmin)
                c = np.sqrt(gamma*p/rho)

                # Sum of Wave speed * surface area
                sum_lamdf = 0.0
                for jdx in range(nface):
                    # Wave speed abs(Vn) + c + max(4/3 \gamma)/rho/(mu/pr+mut/prt)/length
                    lamdf = abs(dot(u[:, idx], svec[jdx, idx], ndims, 1))/rho + c
                    lamdf += (1/rho*max(4/3, gamma)*(mu[idx]/pr + mut[idx]/prt)*
                              smag[jdx, idx]/vol[idx])
                    sum_lamdf += lamdf*smag[jdx, idx]

                # Time step : CFL * vol / sum(lambda_f S_f)
                dt[idx] = cfl*vol[idx] / sum_lamdf

        return self.be.make_loop(self.neles, timestep)

    def _make_recon(self):
        nface, ndims = self.nface, self.ndims
        nvars, nfvars = self.nvars, self.nfvars
        op = self.dxf

        def _cal_recon(i_begin, i_end, upts, grad, lim, fpts):
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

        return self.be.make_loop(self.neles,_cal_recon)

    def _make_div_upts(self):
        nvars, nface = self.nvars, self.nface

        rcp_vol = self.rcp_vol
        ydist = self.ydist

        turb_src = self.turb_src_container()

        def _div_upts(i_begin, i_end, rhs, fpts, upts, grad, dsrc, mu, mut, t=0):
            for idx in range(i_begin, i_end):
                rcp_voli = rcp_vol[idx]
                for jdx in range(nvars):
                    tmp = 0.0
                    for kdx in range(nface):
                        tmp += fpts[kdx, jdx, idx]

                    rhs[jdx, idx] = -rcp_voli*tmp

                # Turbulence source term
                turb_src(upts[:, idx], grad[:, :, idx], mu[idx], mut[idx],
                         ydist[idx], rhs[:, idx], dsrc[:, idx])

        return self.be.make_loop(self.neles, _div_upts)

    def _make_post(self):
        # Get post-process function
        _fix_nonPys = self.fix_nonPys_container()
        _compute_mu = self.mu_container()
        _compute_mut = self.mut_container()

        ydist = self.ydist
        muf = self._const['mu']

        def post(i_begin, i_end, upts, grad, mu, mut):
            # Apply the function over eleemnts
            for idx in range(i_begin, i_end):
                _fix_nonPys(upts[:, idx])
                mu[idx] = _compute_mu(upts[:, idx])
                mut[idx] = _compute_mut(
                    upts[:, idx], grad[:,:,idx], mu[idx], ydist[idx]
                )

                # Limit for turbulence viscosity
                mut[idx] = min(mut[idx], 100000*muf)

        return self.be.make_loop(self.neles, post)