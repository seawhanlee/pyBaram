# -*- coding: utf-8 -*-
from pybaram.solvers.baseadvecdiff import BaseAdvecDiffElements
from pybaram.backends.types import Kernel
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
        self._ydist = ydist = self._wall_distance(btri)

        # Parse initial condition from expressions
        subs = dict(zip('xyz', xc))
        subs.update({'ydist' : ydist})
        ics = [npeval(self.cfg.getexpr('soln-ics', v, self._const), subs)
               for v in self.primevars]
        ics = self.prim_to_conv(ics, self.cfg)

        # Allocate numpy array and copy parsed values
        self._ics = np.empty((self.nvars, self.neles))
        for i in range(self.nvars):
            self._ics[i] = ics[i]        

    def construct_kernels(self, vertex, nreg, impl_op):
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
            self.aux[0] = self._ydist
            delattr(self, '_ydist')
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

    def _wall_distance(self, btri):
        wall_dist = np.empty(self.neles)

        # Define wall distance function
        if self.ndims == 2:
            from pybaram.utils.nb import dist2d_at

            def distf(i_begin, i_end, is_masked, idx, xw, xc, wdist):       
                for _i in range(i_begin, i_end):
                    # Cell index
                    k = is_masked[_i]
                    for _j in range(5):
                        # Candidates of nearest wall index
                        j = idx[_i, _j]
                        status, distj = dist2d_at(xw[j][0], xw[j][1], xc[k])

                        if _j == 0:
                            dist = distj
                        else:
                            dist = min(dist, distj)

                        if status == 0:
                            break

                    # Update wall distance
                    wdist[k] = dist

        elif self.ndims == 3:
            from pybaram.utils.nb import dist3d_at

            def distf(i_begin, i_end, is_masked, idx, xw, xc, wdist):       
                nj = idx.shape[1]
                for _i in range(i_begin, i_end):
                    # Cell index
                    k = is_masked[_i]
                    for _j in range(nj):
                        # Candidates of nearest wall index
                        j = idx[_i, _j]

                        status, distj = dist3d_at(xw[j][0], xw[j][1], xw[j][2], xc[k])

                        if _j == 0:
                            dist = distj
                        else:
                            dist = min(dist, distj)

                        if status == 0:
                            break
                    
                    # Update wall distance
                    wdist[k] = dist

        # Compute wall distance using KDtree version 
        try:
            # pykdtree
            self._wall_distance_kdtree_pykdtree(btri, wall_dist, distf)
        except:
            # Scipy
            self._wall_distance_kdtree_scipy(btri, wall_dist, distf)

        return wall_dist
    
    def _wall_distance_kdtree_scipy(self, xw, wdist, distf):
        from scipy.spatial import KDTree

        xwc = np.average(xw, axis=1)
        
        # Build Tree data
        tree = KDTree(xwc)

        # Check multi-thread or not
        if self.be.multithread == 'single':
            workers = 1
        else:
            workers = -1

        # Fast distance
        fast_distance, fast_idx = tree.query(self.xc, workers=workers)
        wdist[:] = fast_distance

        # Threshold : two times of max distance btw vertex to center of triangle
        threshold = 2*np.max(np.linalg.norm(xw - xwc[:, None], axis=2), axis=1)
        
        # Mask if dist < threshold of nearest triangle
        mask = fast_distance < threshold[fast_idx]
        
        # Detail check (User tunable)
        n_neighbor = max(len(xwc) // 1000, 50)
        _, idx = tree.query(self.xc[mask], k=n_neighbor, workers=workers)

        is_masked = np.where(mask)[0]
        self.be.make_loop(len(is_masked), distf)(is_masked, idx, xw, self.xc, wdist)
        
        # Delete tree
        del(tree)

    def _wall_distance_kdtree_pykdtree(self, xw, wdist, distf):
        from pykdtree.kdtree import KDTree

        xwc = np.average(xw, axis=1)
        
        # Build Tree data
        tree = KDTree(xwc)

        # Fast distance
        fast_distance, fast_idx = tree.query(self.xc)
        wdist[:] = fast_distance

        # Threshold : two times of max distance btw vertex and center of triangle
        threshold = 2*np.max(np.linalg.norm(xw - xwc[:, None], axis=2), axis=1)
        
        # Mask if dist < threshold of nearest triangle
        mask = fast_distance < threshold[fast_idx]
        
        # Detail check (User tunable)
        n_neighbor = max(len(xwc) // 1000, 50)
        _, idx = tree.query(self.xc[mask], k=n_neighbor)

        is_masked = np.where(mask)[0]
        self.be.make_loop(len(is_masked), distf)(is_masked, idx, xw, self.xc, wdist)
        
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