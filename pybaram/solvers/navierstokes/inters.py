# -*- coding: utf-8 -*-
from pybaram.solvers.baseadvecdiff import BaseAdvecDiffIntInters, BaseAdvecDiffBCInters, BaseAdvecDiffMPIInters
from pybaram.backends.types import Kernel
from pybaram.solvers.euler.rsolvers import get_rsolver
from pybaram.solvers.navierstokes.bcs import get_bc
from pybaram.solvers.navierstokes.visflux import make_visflux

import numpy as np


class NavierStokesIntInters(BaseAdvecDiffIntInters):
    def construct_kernels(self, elemap, impl_op):
        super().construct_kernels(elemap)

        # Save viscosity on face (for implicit operator)
        self.muf = muf = self.be.alloc_array((self.nfpts,))

        # Collect face point array
        fpts, gradf = self._fpts, self._gradf

        if impl_op == 'spectral-radius':
            # Collect array to save spectral raidus
            fspr = tuple(cell.fspr for cell in elemap.values())
            self.compute_flux = Kernel(*self._make_flux(impl_op), muf, gradf, fpts, fspr)
        elif impl_op == 'approx-jacobian':
            # Collect array to save Jacobian
            fjmat = tuple(cell.jmat for cell in elemap.values())
            self.compute_flux = Kernel(*self._make_flux(impl_op), muf, gradf, fpts, fjmat)
        else:
            self.compute_flux = Kernel(*self._make_flux(impl_op), muf, gradf, fpts)

    def _make_flux(self, impl_op):
        ndims, nfvars = self.ndims, self.nfvars

        # Constant arrays
        lidx = self.lidx
        ridx = self.ridx
        nf, sf = self.vec_snorm, self.mag_snorm

        # Compiler arguments
        array = self.be.local()
        cplargs = {
            'flux' : self.ele0.flux_container(),
            'to_primevars' : self.ele0.to_flow_primevars(),
            'ndims' : ndims,
            'nfvars' : nfvars,
            'array' : array,
            **self._const
        }

        # Get numerical schems from `rsolvers.py`
        scheme = self.cfg.get('solver', 'riemann-solver')
        flux = get_rsolver(scheme, self.be, cplargs)

        # Get compiled function of viscosity and viscous flux
        compute_mu = self.ele0.mu_container()
        visflux = make_visflux(self.be, cplargs)

        if impl_op == 'spectral-radius':
            # reciprocal of distance between two cells
            rcp_dx = self.rcp_dx

            wave_speed = self.ele0.make_wave_speed()

            def comm_flux_spr(i_begin, i_end, lidx, ridx, nf, sf, rcp_dx, muf, gradf, uf, lam):
                for idx in range(i_begin, i_end):
                    fn = array((nfvars,), np.float64)
                    um = array((nfvars,), np.float64)

                    # Normal vector
                    nfi = nf[:, idx]
                    rcp_dxi = rcp_dx[idx]

                    # Left and right solutions
                    lti, lei, lfi = lidx[:, idx]
                    rti, rei, rfi = ridx[:, idx]
                    ul = uf[lti][lfi, :, lei]
                    ur = uf[rti][rfi, :, rei]
                    
                    # Gradient and solution at face
                    gf = gradf[:, :, idx]

                    for jdx in range(nfvars):
                        um[jdx] = 0.5*(ul[jdx] + ur[jdx])

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)

                    # Compute viscosity and viscous flux
                    muf[idx] = mu = compute_mu(um)
                    visflux(um, gf, nfi, mu, fn)

                    # Compute wave speed on both cell
                    laml = wave_speed(ul, nfi, rcp_dxi, mu)
                    lamr = wave_speed(ur, nfi, rcp_dxi, mu)

                    # Compute spectral radius on face
                    lami = max(laml, lamr)
                    lam[lti][lfi, lei] = lami
                    lam[rti][rfi, rei] = lami

                    for jdx in range(nfvars):
                        # Save it at left and right solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]
                        uf[rti][rfi, jdx, rei] = -fn[jdx]*sf[idx]

            return self.be.make_loop(self.nfpts, comm_flux_spr, lidx, ridx, nf, sf, rcp_dx)
        elif impl_op == 'approx-jacobian':
            from pybaram.solvers.euler.jacobian import make_convective_jacobian
            from pybaram.solvers.navierstokes.jacobian import get_viscous_jacobian

            vistype = self.cfg.get('solver-time-integrator', 'visflux-jacobian', 'tlns')

            # Get Jacobian functions
            pos_jacobian = make_convective_jacobian(self.be, cplargs, 'positive')
            neg_jacobian = make_convective_jacobian(self.be, cplargs, 'negative')
            vis_pos_jacobian = get_viscous_jacobian(vistype, self.be, cplargs, 'positive')
            vis_neg_jacobian = get_viscous_jacobian(vistype, self.be, cplargs, 'negative')

            # reciprocal of distance between two cells
            rcp_dx = self.rcp_dx

            def comm_flux_ajac(i_begin, i_end, lidx, ridx, nf, sf, rcp_dx, muf, gradf, uf, jmats):
                for idx in range(i_begin, i_end):
                    fn = array((nfvars,), np.float64)
                    um = array((nfvars,), np.float64)

                    # Jacobian matrix
                    ap = array((nfvars, nfvars), np.float64)
                    am = array((nfvars, nfvars), np.float64)

                    # Normal vector
                    nfi = nf[:, idx]
                    rcp_dxi = rcp_dx[idx]

                    # Left and right solutions
                    lti, lei, lfi = lidx[:, idx]
                    rti, rei, rfi = ridx[:, idx]
                    ul = uf[lti][lfi, :, lei]
                    ur = uf[rti][rfi, :, rei]
                    
                    # Gradient and solution at face
                    gf = gradf[:, :, idx]

                    for jdx in range(nfvars):
                        um[jdx] = 0.5*(ul[jdx] + ur[jdx])

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)

                    # Compute viscosity and viscous flux
                    muf[idx] = mu = compute_mu(um)
                    visflux(um, gf, nfi, mu, fn)

                    # Compute Jacobian matrix on surface
                    # based on left/right cell
                    pos_jacobian(ul, nfi, ap)
                    neg_jacobian(ur, nfi, am)

                    vis_pos_jacobian(ul, nfi, ap, mu, rcp_dxi)
                    vis_neg_jacobian(ur, nfi, am, mu, rcp_dxi)

                    # Compute approximate Jacobian on face
                    for row in range(nfvars):
                        for col in range(nfvars):
                            jmats[lti][0, row, col, lfi, lei] = ap[row][col]
                            jmats[lti][1, row, col, lfi, lei] = am[row][col]
                            jmats[rti][0, row, col, rfi, rei] = -am[row][col]
                            jmats[rti][1, row, col, rfi, rei] = -ap[row][col]

                    for jdx in range(nfvars):
                        # Save it at left and right solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]
                        uf[rti][rfi, jdx, rei] = -fn[jdx]*sf[idx]

            return self.be.make_loop(self.nfpts, comm_flux_ajac, lidx, ridx, nf, sf, rcp_dx)
        else:
            def comm_flux(i_begin, i_end, lidx, ridx, nf, sf, muf, gradf, uf):
                for idx in range(i_begin, i_end):
                    fn = array((nfvars,), np.float64)
                    um = array((nfvars,), np.float64)

                    # Normal vector
                    nfi = nf[:, idx]

                    # Left and right solutions
                    lti, lei, lfi = lidx[:, idx]
                    rti, rei, rfi = ridx[:, idx]
                    ul = uf[lti][lfi, :, lei]
                    ur = uf[rti][rfi, :, rei]
                    
                    # Gradient and solution at face
                    gf = gradf[:, :, idx]

                    for jdx in range(nfvars):
                        um[jdx] = 0.5*(ul[jdx] + ur[jdx])

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)

                    # Compute viscosity and viscous flux
                    muf[idx] = mu = compute_mu(um)
                    visflux(um, gf, nfi, mu, fn)

                    for jdx in range(nfvars):
                        # Save it at left and right solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]
                        uf[rti][rfi, jdx, rei] = -fn[jdx]*sf[idx]

            return self.be.make_loop(self.nfpts, comm_flux, lidx, ridx, nf, sf)


class NavierStokesMPIInters(BaseAdvecDiffMPIInters):
    def construct_kernels(self, elemap, impl_op):
        super().construct_kernels(elemap)

        # Save viscosity on face (for implicit operator)
        self.muf = muf = self.be.alloc_array((self.nfpts,))

        # Collect face point array
        fpts, gradf = self._fpts, self._gradf
        rhs = self._rhs

        if impl_op == 'spectral-radius':
            # Collect array to save spectral raidus
            fspr = tuple(cell.fspr for cell in elemap.values())
            self.compute_flux = Kernel(*self._make_flux(impl_op), muf, gradf, rhs, fpts, fspr)
        elif impl_op == 'approx-jacobian':
            # Collect array to save Jacobian
            fjmat = tuple(cell.jmat for cell in elemap.values())
            self.compute_flux = Kernel(*self._make_flux(impl_op), muf, gradf, rhs, fpts, fjmat)
        else:
            self.compute_flux = Kernel(*self._make_flux(impl_op), muf, gradf, rhs, fpts)

    def _make_flux(self, impl_op):
        ndims, nfvars = self.ndims, self.nfvars
        lidx = self.lidx
        nf, sf = self.vec_snorm, self.mag_snorm

        # Compiler arguments
        array = self.be.local()
        cplargs = {
            'flux' : self.ele0.flux_container(),
            'to_primevars' : self.ele0.to_flow_primevars(),
            'ndims' : ndims,
            'nfvars' : nfvars,
            'array' : array,
            **self._const
        }

        # Get numerical schems from `rsolvers.py`
        scheme = self.cfg.get('solver', 'riemann-solver')
        flux = get_rsolver(scheme, self.be, cplargs)

        # Get compiled function of viscosity and viscous flux
        compute_mu = self.ele0.mu_container()
        visflux = make_visflux(self.be, cplargs)

        if impl_op == 'spectral-radius':
            # reciprocal of distance between two cells
            rcp_dx = self.rcp_dx

            # Get wave speed function
            wave_speed = self.ele0.make_wave_speed()

            def comm_flux_spr(i_begin, i_end, lidx, nf, sf, rcp_dx, muf, gradf, rhs, uf, lam):
                for idx in range(i_begin, i_end):
                    fn = array((nfvars,), np.float64)
                    um = array((nfvars,), np.float64)

                    # Normal vector
                    nfi = nf[:, idx]
                    rcp_dxi = rcp_dx[idx]

                    # Left and right solutions
                    lti, lei, lfi = lidx[:, idx]
                    ul = uf[lti][lfi, :, lei]
                    ur = rhs[:, idx]

                    # Gradient and solution at face
                    gf = gradf[:, :, idx]

                    for jdx in range(nfvars):
                        um[jdx] = 0.5*(ul[jdx] + ur[jdx])

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)
                    
                    # Compute viscosity and viscous flux
                    muf[idx] = mu = compute_mu(um)
                    visflux(um, gf, nfi, mu, fn)

                    # Compute spectral radius on face
                    lami = wave_speed(ul, nfi, rcp_dxi, mu)
                    lam[lti][lfi, lei] = lami

                    for jdx in range(nfvars):
                        # Save it at left solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

            return self.be.make_loop(self.nfpts, comm_flux_spr, lidx, nf, sf, rcp_dx)
        elif impl_op == 'approx-jacobian':
            from pybaram.solvers.euler.jacobian import make_convective_jacobian
            from pybaram.solvers.navierstokes.jacobian import get_viscous_jacobian

            vistype = self.cfg.get('solver-time-integrator', 'visflux-jacobian', 'tlns')

            # Get Jacobian functions
            pos_jacobian = make_convective_jacobian(self.be, cplargs, 'positive')
            vis_jacobian = get_viscous_jacobian(vistype, self.be, cplargs)

            # reciprocal of distance between two cells
            rcp_dx = self.rcp_dx

            def comm_flux_ajac(i_begin, i_end, lidx, nf, sf, rcp_dx, muf, gradf, rhs, uf, jmats):
                for idx in range(i_begin, i_end):
                    fn = array((nfvars,), np.float64)
                    um = array((nfvars,), np.float64)

                    # Jacobian matrix
                    ap = array((nfvars, nfvars), np.float64)

                    # Normal vector
                    nfi = nf[:, idx]
                    rcp_dxi = rcp_dx[idx]

                    # Left and right solutions
                    lti, lei, lfi = lidx[:, idx]
                    ul = uf[lti][lfi, :, lei]
                    ur = rhs[:, idx]

                    # Gradient and solution at face
                    gf = gradf[:, :, idx]

                    for jdx in range(nfvars):
                        um[jdx] = 0.5*(ul[jdx] + ur[jdx])

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)
                    
                    # Compute viscosity and viscous flux
                    muf[idx] = mu = compute_mu(um)
                    visflux(um, gf, nfi, mu, fn)

                    # Compute Jacobian matrix on surface
                    pos_jacobian(ul, nfi, ap)
                    vis_jacobian(ul, nfi, ap, mu, rcp_dxi)

                    # Compute approximate Jacobian on face
                    for row in range(nfvars):
                        for col in range(nfvars):
                            jmats[lti][0, row, col, lfi, lei] = ap[row][col]

                    for jdx in range(nfvars):
                        # Save it at left solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

            return self.be.make_loop(self.nfpts, comm_flux_ajac, lidx, nf, sf, rcp_dx)
        else:        
            def comm_flux(i_begin, i_end, lidx, nf, sf, muf, gradf, rhs, uf):
                for idx in range(i_begin, i_end):
                    fn = array((nfvars,), np.float64)
                    um = array((nfvars,), np.float64)

                    # Normal vector
                    nfi = nf[:, idx]

                    # Left and right solutions
                    lti, lei, lfi = lidx[:, idx]
                    ul = uf[lti][lfi, :, lei]
                    ur = rhs[:, idx]

                    # Gradient and solution at face
                    gf = gradf[:, :, idx]

                    for jdx in range(nfvars):
                        um[jdx] = 0.5*(ul[jdx] + ur[jdx])

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)
                    
                    # Compute viscosity and viscous flux
                    muf[idx] = mu = compute_mu(um)
                    visflux(um, gf, nfi, mu, fn)

                    for jdx in range(nfvars):
                        # Save it at left solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

            return self.be.make_loop(self.nfpts, comm_flux, lidx, nf, sf)


class NavierStokesBCInters(BaseAdvecDiffBCInters):
    _get_bc = get_bc

    def construct_kernels(self, elemap, impl_op):
        super().construct_kernels(elemap)
        
        # Save viscosity on face (for implicit operator)
        self.muf = muf = self.be.alloc_array((self.nfpts,))

        # Collect face point array
        fpts, gradf = self._fpts, self._gradf

        if impl_op == 'spectral-radius':
            # Collect array to save spectral raidus
            fspr = tuple(cell.fspr for cell in elemap.values())
            self.compute_flux = Kernel(*self._make_flux(impl_op), muf, gradf, fpts, fspr)
        elif impl_op == 'approx-jacobian':
            # Collect array to save Jacobian
            fjmat = tuple(cell.jmat for cell in elemap.values())
            self.compute_flux = Kernel(*self._make_flux(impl_op), muf, gradf, fpts, fjmat)
        else:
            self.compute_flux = Kernel(*self._make_flux(impl_op), muf, gradf, fpts)

    def _make_flux(self, impl_op):
        ndims, nfvars = self.ndims, self.nfvars
        lidx = self.lidx
        nf, sf = self.vec_snorm, self.mag_snorm

        # Compiler arguments
        array = self.be.local()
        cplargs = {
            'flux' : self.ele0.flux_container(),
            'to_primevars' : self.ele0.to_flow_primevars(),
            'ndims' : ndims,
            'nfvars' : nfvars,
            'array' : array,
            **self._const
        }

        # Get numerical schems from `rsolvers.py`
        scheme = self.cfg.get('solver', 'riemann-solver')
        flux = get_rsolver(scheme, self.be, cplargs)

        # Get compiled function of viscosity and viscous flux
        compute_mu = self.ele0.mu_container()
        visflux = make_visflux(self.be, cplargs)

        # Get bc function (`self.bc` was defined at `baseadvec.inters`)
        bc = self.bc
        if impl_op == 'spectral-radius':
            # reciprocal of distance between two cells
            rcp_dx = self.rcp_dx

            wave_speed = self.ele0.make_wave_speed()
            
            def comm_flux_spr(i_begin, i_end, lidx, nf, sf, rcp_dx, muf, gradf, uf, lam):
                for idx in range(i_begin, i_end):
                    ur = array((nfvars,), np.float64)
                    um = array((nfvars,), np.float64)
                    fn = array((nfvars,), np.float64)

                    # Normal vector
                    nfi = nf[:, idx]
                    rcp_dxi = rcp_dx[idx]

                    # Left solutions
                    lti, lei, lfi = lidx[:, idx]
                    ul = uf[lti][lfi, :, lei]

                    # Gradient at face
                    gf = gradf[:, :, idx]

                    # Compute BC
                    bc(ul, ur, nfi)

                    # Solution at face
                    for jdx in range(nfvars):
                        um[jdx] = 0.5*(ul[jdx] + ur[jdx])

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)
                    
                    # Compute viscosity and viscous flux
                    muf[idx] = mu = compute_mu(um)
                    visflux(um, gf, nfi, mu, fn)

                    # Compute spectral radius on face
                    lami = wave_speed(ul, nfi, rcp_dxi, mu)
                    lam[lti][lfi, lei] = lami

                    for jdx in range(nfvars):
                        # Save it at left solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

            return self.be.make_loop(self.nfpts, comm_flux_spr, lidx, nf, sf, rcp_dx)
        elif impl_op == 'approx-jacobian':
            from pybaram.solvers.euler.jacobian import make_convective_jacobian
            from pybaram.solvers.navierstokes.jacobian import get_viscous_jacobian

            vistype = self.cfg.get('solver-time-integrator', 'visflux-jacobian', 'tlns')

            # Get Jacobian functions
            pos_jacobian = make_convective_jacobian(self.be, cplargs, 'positive')
            vis_jacobian = get_viscous_jacobian(vistype, self.be, cplargs)

            # reciprocal of distance between two cells
            rcp_dx = self.rcp_dx

            def comm_flux_ajac(i_begin, i_end, lidx, nf, sf, rcp_dx, muf, gradf, uf, jmats):
                for idx in range(i_begin, i_end):
                    ur = array((nfvars,), np.float64)
                    um = array((nfvars,), np.float64)
                    fn = array((nfvars,), np.float64)

                    # Jacobian matrix
                    ap = array((nfvars, nfvars), np.float64)

                    # Normal vector
                    nfi = nf[:, idx]
                    rcp_dxi = rcp_dx[idx]

                    # Left solutions
                    lti, lei, lfi = lidx[:, idx]
                    ul = uf[lti][lfi, :, lei]

                    # Gradient at face
                    gf = gradf[:, :, idx]

                    # Compute BC
                    bc(ul, ur, nfi)

                    # Solution at face
                    for jdx in range(nfvars):
                        um[jdx] = 0.5*(ul[jdx] + ur[jdx])

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)
                    
                    # Compute viscosity and viscous flux
                    muf[idx] = mu = compute_mu(um)
                    visflux(um, gf, nfi, mu, fn)

                    # Compute Jacobian matrix on surface
                    pos_jacobian(ul, nfi, ap)
                    vis_jacobian(ul, nfi, ap, mu, rcp_dxi)

                    # Compute approximate Jacobian on face
                    for row in range(nfvars):
                        for col in range(nfvars):
                            jmats[lti][0, row, col, lfi, lei] = ap[row][col]

                    for jdx in range(nfvars):
                        # Save it at left solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

            return self.be.make_loop(self.nfpts, comm_flux_ajac, lidx, nf, sf, rcp_dx)
        else:
            def comm_flux(i_begin, i_end, lidx, nf, sf, muf, gradf, uf):
                for idx in range(i_begin, i_end):
                    ur = array((nfvars,), np.float64)
                    um = array((nfvars,), np.float64)
                    fn = array((nfvars,), np.float64)

                    # Normal vector
                    nfi = nf[:, idx]

                    # Left solutions
                    lti, lei, lfi = lidx[:, idx]
                    ul = uf[lti][lfi, :, lei]

                    # Gradient at face
                    gf = gradf[:, :, idx]

                    # Compute BC
                    bc(ul, ur, nfi)

                    # Solution at face
                    for jdx in range(nfvars):
                        um[jdx] = 0.5*(ul[jdx] + ur[jdx])

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)
                    
                    # Compute viscosity and viscous flux
                    muf[idx] = mu = compute_mu(um)
                    visflux(um, gf, nfi, mu, fn)

                    for jdx in range(nfvars):
                        # Save it at left solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

        return self.be.make_loop(self.nfpts, comm_flux, lidx, nf, sf)

    def _make_spec_rad(self):
        lidx = self.lidx
        nf = self.vec_snorm
        
        # reciprocal of distance between two cells
        rcp_dx = self.rcp_dx

        wave_speed = self.ele0.make_wave_speed()

        def comm_spr(i_begin, i_end, lidx, nf, rcp_dx, muf, uf, lam):
            for idx in range(i_begin, i_end):
                # Normal vector
                nfi = nf[:, idx]
                rcp_dxi = rcp_dx[idx]

                # Left solution
                lti, lei, lfi = lidx[:, idx]
                ul = uf[lti][lfi, :, lei]

                # Get viscosity on face (saved at rhside)
                mu = muf[idx]

                # Compute spectral radius on face
                lami = wave_speed(ul, nfi, rcp_dxi, mu)
                lam[lti][lfi, lei] = lami

        return self.be.make_loop(self.nfpts, comm_spr, lidx, nf, rcp_dx)

    def _make_aprx_jac(self):
        from pybaram.solvers.euler.jacobian import make_convective_jacobian
        from pybaram.solvers.navierstokes.jacobian import get_viscous_jacobian

        nfvars = self.nfvars
        lidx = self.lidx
        nf = self.vec_snorm

        cplargs = {
            'ndims': self.ndims,
            'nfvars': self.nfvars,
            'gamma': self.ele0._const['gamma'],
            'pr': self.ele0._const['pr'],
            'to_prim': self.ele0.to_flow_primevars()
        }

        vistype = self.cfg.get('solver-time-integrator', 'visflux-jacobian', 'tlns')

        # Get Jacobian functions
        pos_jacobian = make_convective_jacobian(self.be, cplargs, 'positive')
        vis_jacobian = get_viscous_jacobian(vistype, self.be, cplargs)

        # reciprocal of distance between two cells
        rcp_dx = self.rcp_dx

        # Temporal matrix
        array = self.be.local()

        def comm_apj(i_begin, i_end, lidx, nf, rcp_dx, muf, uf, jmats):
            for idx in range(i_begin, i_end):
                # Jacobian matrix
                ap = array((nfvars, nfvars), np.float64)

                # Normal vector
                nfi = nf[:, idx]
                rcp_dxi = rcp_dx[idx]

                # Left and right solutions
                lti, lei, lfi = lidx[:, idx]
                ul = uf[lti][lfi, :, lei]

                # Get viscosity on face (saved at rhside)
                mu = muf[idx]

                # Compute Jacobian matrix on surface
                pos_jacobian(ul, nfi, ap)
                vis_jacobian(ul, nfi, ap, mu, rcp_dxi)

                # Compute approximate Jacobian on face
                for row in range(nfvars):
                    for col in range(nfvars):
                        jmats[lti][0, row, col, lfi, lei] = ap[row][col]

        return self.be.make_loop(self.nfpts, comm_apj, lidx, nf, rcp_dx)


class NavierStokesSlipWallBCInters(NavierStokesBCInters):
    name = 'slip-wall'


class NavierStokesAdiaWallBCInters(NavierStokesBCInters):
    name = 'adia-wall'


class NavierStokesIsothermWallBCInters(NavierStokesBCInters):
    name = 'isotherm-wall'
    _reqs = ['cptw']


class NavierStokesSupOutBCInters(NavierStokesBCInters):
    name = 'sup-out'


class NavierStokesSupInBCInters(NavierStokesBCInters):
    name = 'sup-in'

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)

        self._reqs = self.primevars


class NavierStokesFarInBCInters(NavierStokesBCInters):
    name = 'far'

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)

        self._reqs = self.primevars


class NavierStokesSubOutPBCInters(NavierStokesBCInters):
    name = 'sub-outp'
    _reqs = ['p']


class NavierStokesSubInvBCInters(NavierStokesBCInters):
    name = 'sub-inv'

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)

        self._reqs = ['rho'] + ['u', 'v', 'w'][:self.ndims]


class NavierStokesSubInpttBCInters(NavierStokesBCInters):
    name = 'sub-inptt'

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)

        self._reqs = ['p0', 'cpt0', 'dir']
