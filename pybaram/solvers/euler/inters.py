# -*- coding: utf-8 -*-
from pybaram.solvers.baseadvec import BaseAdvecIntInters, BaseAdvecBCInters, BaseAdvecMPIInters
from pybaram.backends.types import Kernel
from pybaram.solvers.euler.rsolvers import get_rsolver
from pybaram.solvers.euler.bcs import get_bc

import numpy as np


class EulerIntInters(BaseAdvecIntInters):
    def construct_kernels(self, elemap, impl_op):
        super().construct_kernels(elemap)        

        # Collect face point array
        fpts = self._fpts

        if impl_op == 'spectral-radius':
            # Collect array to save spectral raidus
            fspr = tuple(cell.fspr for cell in elemap.values())
            self.compute_flux = Kernel(self._make_flux(impl_op), fpts, fspr)
        elif impl_op == 'approx-jacobian':
            # Collect array to save Jacobian
            fjmat = tuple(cell.jmat for cell in elemap.values())
            self.compute_flux = Kernel(self._make_flux(impl_op), fpts, fjmat)
        else:
            self.compute_flux = Kernel(self._make_flux(impl_op), fpts)

    def _make_flux(self, impl_op):
        ndims, nfvars = self.ndims, self.nfvars
        lt, le, lf = self._lidx
        rt, re, rf = self._ridx
        nf, sf = self._vec_snorm, self._mag_snorm

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

        if impl_op == 'spectral-radius':
            # Get wave speed function
            wave_speed = self.ele0.make_wave_speed()

            def comm_flux_spr(i_begin, i_end, uf, lam):
                for idx in range(i_begin, i_end):
                    fn = array((nfvars,))

                    # Normal vector
                    nfi = nf[:, idx]

                    # Left and right solutions
                    lti, lfi, lei = lt[idx], lf[idx], le[idx]
                    rti, rfi, rei = rt[idx], rf[idx], re[idx]
                    ul = uf[lti][lfi, :, lei]
                    ur = uf[rti][rfi, :, rei]

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)

                    # Compute wave speed on both cell
                    laml = wave_speed(ul, nfi)
                    lamr = wave_speed(ur, nfi)

                    # Compute spectral radius on face
                    lami = max(laml, lamr)
                    lam[lti][lfi, lei] = lami
                    lam[rti][rfi, rei] = lami

                    for jdx in range(nfvars):
                        # Save it at left and right solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]
                        uf[rti][rfi, jdx, rei] = -fn[jdx]*sf[idx]

            return self.be.make_loop(self.nfpts, comm_flux_spr)
        elif impl_op == 'approx-jacobian':
            from pybaram.solvers.euler.jacobian import make_convective_jacobian

            # Get Jacobian functions
            pos_jacobian = make_convective_jacobian(self.be, cplargs, 'positive')
            neg_jacobian = make_convective_jacobian(self.be, cplargs, 'negative')

            def comm_flux_ajac(i_begin, i_end, uf, jmats):
                for idx in range(i_begin, i_end):
                    fn = array((nfvars,))

                    # Jacobian matrix
                    ap = array((nfvars, nfvars))
                    am = array((nfvars, nfvars))

                    # Normal vector
                    nfi = nf[:, idx]

                    # Left and right solutions
                    lti, lfi, lei = lt[idx], lf[idx], le[idx]
                    rti, rfi, rei = rt[idx], rf[idx], re[idx]
                    ul = uf[lti][lfi, :, lei]
                    ur = uf[rti][rfi, :, rei]

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)

                    # Compute Jacobian matrix on surface
                    # based on left/right cell
                    pos_jacobian(ul, nfi, ap)
                    neg_jacobian(ur, nfi, am)

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

            return self.be.make_loop(self.nfpts, comm_flux_ajac)
        else:            
            def comm_flux(i_begin, i_end, uf):
                for idx in range(i_begin, i_end):
                    fn = array((nfvars,))

                    # Normal vector
                    nfi = nf[:, idx]

                    # Left and right solutions
                    lti, lfi, lei = lt[idx], lf[idx], le[idx]
                    rti, rfi, rei = rt[idx], rf[idx], re[idx]
                    ul = uf[lti][lfi, :, lei]
                    ur = uf[rti][rfi, :, rei]

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)

                    for jdx in range(nfvars):
                        # Save it at left and right solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]
                        uf[rti][rfi, jdx, rei] = -fn[jdx]*sf[idx]

            return self.be.make_loop(self.nfpts, comm_flux)    


class EulerMPIInters(BaseAdvecMPIInters):
    def construct_kernels(self, elemap, impl_op):
        super().construct_kernels(elemap)        

        # Collect face point array and buffer
        fpts, rhs = self._fpts, self._rhs

        if impl_op == 'spectral-radius':
            # Collect array to save spectral raidus
            fspr = tuple(cell.fspr for cell in elemap.values())
            self.compute_flux = Kernel(self._make_flux(impl_op), rhs, fpts, fspr)
        elif impl_op == 'approx-jacobian':
            # Collect array to save Jacobian
            fjmat = tuple(cell.jmat for cell in elemap.values())
            self.compute_flux = Kernel(self._make_flux(impl_op), rhs, fpts, fjmat)
        else:
            self.compute_flux = Kernel(self._make_flux(impl_op), rhs, fpts)


    def _make_flux(self, impl_op):
        ndims, nfvars = self.ndims, self.nfvars
        lt, le, lf = self._lidx
        nf, sf = self._vec_snorm, self._mag_snorm

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

        if impl_op == 'spectral-radius':
            # Get wave speed function
            wave_speed = self.ele0.make_wave_speed()

            def comm_flux_spr(i_begin, i_end, rhs, uf, lam):
                for idx in range(i_begin, i_end):
                    fn = array((nfvars,))

                    # Normal vector
                    nfi = nf[:, idx]

                    # Left and right solutions
                    lti, lfi, lei = lt[idx], lf[idx], le[idx]
                    ul = uf[lti][lfi, :, lei]
                    ur = rhs[:, idx]

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)

                    # Compute spectral radius on face
                    lami = wave_speed(ul, nfi)
                    lam[lti][lfi, lei] = lami

                    for jdx in range(nfvars):
                        # Save it at left solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

            return self.be.make_loop(self.nfpts, comm_flux_spr)
        elif impl_op == 'approx-jacobian':
            from pybaram.solvers.euler.jacobian import make_convective_jacobian

            # Get Jacobian functions
            com_aprx_jac = make_convective_jacobian(self.be, cplargs, 'positive')

            def comm_flux_ajac(i_begin, i_end, rhs, uf, jmats):
                for idx in range(i_begin, i_end):
                    fn = array((nfvars,))
                    ap = array((nfvars, nfvars))

                    # Normal vector
                    nfi = nf[:, idx]

                    # Left and right solutions
                    lti, lfi, lei = lt[idx], lf[idx], le[idx]
                    ul = uf[lti][lfi, :, lei]
                    ur = rhs[:, idx]

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)

                    # Compute Jacobian matrix on face
                    com_aprx_jac(ul, nfi, ap)
                    for row in range(nfvars):
                        for col in range(nfvars):
                            jmats[lti][0, row, col, lfi, lei] = ap[row][col]

                    for jdx in range(nfvars):
                        # Save it at left solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

            return self.be.make_loop(self.nfpts, comm_flux_ajac)
        else:
            def comm_flux(i_begin, i_end, rhs, uf):
                for idx in range(i_begin, i_end):
                    fn = array((nfvars,))

                    # Normal vector
                    nfi = nf[:, idx]

                    # Left and right solutions
                    lti, lfi, lei = lt[idx], lf[idx], le[idx]
                    ul = uf[lti][lfi, :, lei]
                    ur = rhs[:, idx]

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)

                    for jdx in range(nfvars):
                        # Save it at left solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

            return self.be.make_loop(self.nfpts, comm_flux)


class EulerBCInters(BaseAdvecBCInters):
    _get_bc = get_bc

    def construct_kernels(self, elemap, impl_op):
        super().construct_kernels(elemap)
        
        # Collect face point array
        fpts = self._fpts

        if impl_op == 'spectral-radius':
            # Collect array to save spectral raidus
            fspr = tuple(cell.fspr for cell in elemap.values())
            #self.compute_spec_rad = Kernel(self._make_spec_rad(), fpts, fspr)
            self.compute_flux = Kernel(self._make_flux(impl_op), fpts, fspr)
        elif impl_op == 'approx-jacobian':
            # Collect array to save Jacobian
            fjmat = tuple(cell.jmat for cell in elemap.values())
            self.compute_flux = Kernel(self._make_flux(impl_op), fpts, fjmat)
        else:
            self.compute_flux = Kernel(self._make_flux(impl_op), fpts)

    def _make_flux(self, impl_op):
        ndims, nfvars = self.ndims, self.nfvars

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

        # Get bc function (`self.bc` was defined at `baseadvec.inters`)
        bc = self.bc

        lt, le, lf = self._lidx
        nf, sf = self._vec_snorm, self._mag_snorm,

        if impl_op == 'spectral-radius':
            # Get wave speed function
            wave_speed = self.ele0.make_wave_speed()

            def bc_flux_spr(i_begin, i_end, uf, lam):
                for idx in range(i_begin, i_end):
                    fn = array((nfvars,))
                    ur = array((nfvars,))

                    # Normal vector
                    nfi = nf[:, idx]

                    # Left solutions
                    lti, lfi, lei = lt[idx], lf[idx], le[idx]
                    ul = uf[lti][lfi, :, lei]

                    # Compute BC
                    bc(ul, ur, nfi)

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)
                    
                    # Compute spectral radius on face
                    lami = wave_speed(ul, nfi)
                    lam[lti][lfi, lei] = lami

                    for jdx in range(nfvars):
                        # Save it at left solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

            return self.be.make_loop(self.nfpts, bc_flux_spr)
        elif impl_op == 'approx-jacobian':
            from pybaram.solvers.euler.jacobian import make_convective_jacobian

            # Get Jacobian functions
            pos_jacobian = make_convective_jacobian(self.be, cplargs, 'positive')

            def bc_flux_ajac(i_begin, i_end, uf, jmats):
                for idx in range(i_begin, i_end):
                    fn = array((nfvars,))
                    ur = array((nfvars,))
                    ap = array((nfvars, nfvars))

                    # Normal vector
                    nfi = nf[:, idx]

                    # Left solutions
                    lti, lfi, lei = lt[idx], lf[idx], le[idx]
                    ul = uf[lti][lfi, :, lei]

                    # Compute BC
                    bc(ul, ur, nfi)

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)
                    
                    # Compute Jacobian matrix on face
                    pos_jacobian(ul, nfi, ap)
                    for row in range(nfvars):
                        for col in range(nfvars):
                            jmats[lti][0, row, col, lfi, lei] = ap[row][col]

                    for jdx in range(nfvars):
                        # Save it at left solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

            return self.be.make_loop(self.nfpts, bc_flux_ajac)
        else:
            def bc_flux(i_begin, i_end, uf):
                for idx in range(i_begin, i_end):
                    fn = array((nfvars,))
                    ur = array((nfvars,))

                    # Normal vector
                    nfi = nf[:, idx]

                    # Left solutions
                    lti, lfi, lei = lt[idx], lf[idx], le[idx]
                    ul = uf[lti][lfi, :, lei]

                    # Compute BC
                    bc(ul, ur, nfi)

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)

                    for jdx in range(nfvars):
                        # Save it at left solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

            return self.be.make_loop(self.nfpts, bc_flux)


class EulerSupOutBCInters(EulerBCInters):
    name = 'sup-out'


class EulerSlipWallBCInters(EulerBCInters):
    name = 'slip-wall'


class EulerSupInBCInters(EulerBCInters):
    name = 'sup-in'

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)

        self._reqs = self.primevars


class EulerFarInBCInters(EulerBCInters):
    name = 'far'

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)

        self._reqs = self.primevars


class EulerSubOutPBCInters(EulerBCInters):
    name = 'sub-outp'
    _reqs = ['p']


class EulerSubInvBCInters(EulerBCInters):
    name = 'sub-inv'

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)

        self._reqs = ['rho'] + ['u', 'v', 'w'][:self.ndims]


class EulerSubInpttBCInters(EulerBCInters):
    name = 'sub-inptt'

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)

        self._reqs = ['p0', 'cpt0', 'dir']


class EulerSubOutMdotBCInters(EulerBCInters):
    name = 'sub-outmdot'
    _reqs = ['mdot', 'dir']

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)
