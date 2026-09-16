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
            self.compute_flux = Kernel(
                *self._make_flux(impl_op), fpts, self.rank_fspr
            )
        elif impl_op == 'approx-jacobian':
            self.compute_flux = Kernel(
                *self._make_flux(impl_op), fpts, self.rank_jmat
            )
        else:
            self.compute_flux = Kernel(*self._make_flux(impl_op), fpts)

    def _make_flux(self, impl_op):
        ndims, nfvars = self.ndims, self.nfvars
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

        if impl_op == 'spectral-radius':
            # Get wave speed function
            wave_speed = self.ele0.make_wave_speed()

            def comm_flux_spr(i_begin, i_end, lidx, ridx, nf, sf, face_ids,
                              uf, lam):
                for idx in range(i_begin, i_end):
                    fn = array((nfvars,), np.float64)

                    # Normal vector
                    nfi = nf[:, idx]

                    # Left and right solutions
                    lti, lei, lfi = lidx[:, idx]
                    rti, rei, rfi = ridx[:, idx]
                    ul = uf[lti][lfi, :, lei]
                    ur = uf[rti][rfi, :, rei]

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)

                    # Compute wave speed on both cell
                    laml = wave_speed(ul, nfi)
                    lamr = wave_speed(ur, nfi)

                    # Compute spectral radius on face
                    lami = max(laml, lamr)
                    lam[face_ids[idx]] = lami

                    for jdx in range(nfvars):
                        # Save it at left and right solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]
                        uf[rti][rfi, jdx, rei] = -fn[jdx]*sf[idx]

            return self.be.make_loop(
                self.nfpts, comm_flux_spr, lidx, ridx, nf, sf,
                self.rank_face_ids
            )
        elif impl_op == 'approx-jacobian':
            from pybaram.solvers.euler.jacobian import make_convective_jacobian

            # Get Jacobian functions
            pos_jacobian = make_convective_jacobian(self.be, cplargs, 'positive')
            neg_jacobian = make_convective_jacobian(self.be, cplargs, 'negative')

            def comm_flux_ajac(i_begin, i_end, lidx, ridx, nf, sf, face_ids,
                               uf, jmat):
                for idx in range(i_begin, i_end):
                    fn = array((nfvars,), np.float64)

                    # Jacobian matrix
                    ap = array((nfvars, nfvars), np.float64)
                    am = array((nfvars, nfvars), np.float64)

                    # Normal vector
                    nfi = nf[:, idx]

                    # Left and right solutions
                    lti, lei, lfi = lidx[:, idx]
                    rti, rei, rfi = ridx[:, idx]
                    ul = uf[lti][lfi, :, lei]
                    ur = uf[rti][rfi, :, rei]

                    flux(ul, ur, nfi, fn)

                    # Compute Jacobian matrix on surface
                    # based on left/right cell
                    pos_jacobian(ul, nfi, ap)
                    neg_jacobian(ur, nfi, am)

                    # Compute approximate Jacobian on face
                    for row in range(nfvars):
                        for col in range(nfvars):
                            face = face_ids[idx]
                            jmat[0, row, col, face] = ap[row][col]
                            jmat[1, row, col, face] = am[row][col]

                    for jdx in range(nfvars):
                        # Save it at left and right solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]
                        uf[rti][rfi, jdx, rei] = -fn[jdx]*sf[idx]

            return self.be.make_loop(
                self.nfpts, comm_flux_ajac, lidx, ridx, nf, sf,
                self.rank_face_ids
            )
        else:            
            def comm_flux(i_begin, i_end, lidx, ridx, nf, sf, uf):
                for idx in range(i_begin, i_end):
                    fn = array((nfvars,), np.float64)

                    # Normal vector
                    nfi = nf[:, idx]

                    # Left and right solutions
                    lti, lei, lfi = lidx[:, idx]
                    rti, rei, rfi = ridx[:, idx]
                    ul = uf[lti][lfi, :, lei]
                    ur = uf[rti][rfi, :, rei]

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)

                    for jdx in range(nfvars):
                        # Save it at left and right solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]
                        uf[rti][rfi, jdx, rei] = -fn[jdx]*sf[idx]

            return self.be.make_loop(self.nfpts, comm_flux, lidx, ridx, nf, sf)


class EulerMPIInters(BaseAdvecMPIInters):
    def construct_kernels(self, elemap, impl_op):
        super().construct_kernels(elemap)

        # Collect face point array and buffer
        fpts, rhs = self._fpts, self._rhs

        if impl_op == 'spectral-radius':
            self.compute_flux = Kernel(
                *self._make_flux(impl_op), rhs, fpts, self.rank_fspr
            )
        elif impl_op == 'approx-jacobian':
            self.compute_flux = Kernel(
                *self._make_flux(impl_op), rhs, fpts, self.rank_jmat
            )
        else:
            self.compute_flux = Kernel(*self._make_flux(impl_op), rhs, fpts)


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

        if impl_op == 'spectral-radius':
            # Get wave speed function
            wave_speed = self.ele0.make_wave_speed()

            def comm_flux_spr(i_begin, i_end, lidx, nf, sf, face_ids,
                              rhs, uf, lam):
                for idx in range(i_begin, i_end):
                    fn = array((nfvars,), np.float64)

                    # Normal vector
                    nfi = nf[:, idx]

                    # Left and right solutions
                    lti, lei, lfi = lidx[:, idx]
                    ul = uf[lti][lfi, :, lei]
                    ur = rhs[:, idx]

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)

                    # Compute spectral radius on face
                    lami = wave_speed(ul, nfi)
                    lam[face_ids[idx]] = lami

                    for jdx in range(nfvars):
                        # Save it at left solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

            return self.be.make_loop(
                self.nfpts, comm_flux_spr, lidx, nf, sf,
                self.rank_face_ids
            )
        elif impl_op == 'approx-jacobian':
            from pybaram.solvers.euler.jacobian import make_convective_jacobian

            # Get Jacobian functions
            com_aprx_jac = make_convective_jacobian(self.be, cplargs, 'positive')

            def comm_flux_ajac(i_begin, i_end, lidx, nf, sf, face_ids,
                               rhs, uf, jmat):
                for idx in range(i_begin, i_end):
                    fn = array((nfvars,), np.float64)
                    ap = array((nfvars, nfvars), np.float64)

                    # Normal vector
                    nfi = nf[:, idx]

                    # Left and right solutions
                    lti, lei, lfi = lidx[:, idx]
                    ul = uf[lti][lfi, :, lei]
                    ur = rhs[:, idx]

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)

                    # Compute Jacobian matrix on face
                    com_aprx_jac(ul, nfi, ap)
                    for row in range(nfvars):
                        for col in range(nfvars):
                            jmat[0, row, col, face_ids[idx]] = ap[row][col]

                    for jdx in range(nfvars):
                        # Save it at left solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

            return self.be.make_loop(
                self.nfpts, comm_flux_ajac, lidx, nf, sf,
                self.rank_face_ids
            )
        else:
            def comm_flux(i_begin, i_end, lidx, nf, sf, rhs, uf):
                for idx in range(i_begin, i_end):
                    fn = array((nfvars,), np.float64)

                    # Normal vector
                    nfi = nf[:, idx]

                    # Left and right solutions
                    lti, lei, lfi = lidx[:, idx]
                    ul = uf[lti][lfi, :, lei]
                    ur = rhs[:, idx]

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)

                    for jdx in range(nfvars):
                        # Save it at left solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

            return self.be.make_loop(self.nfpts, comm_flux, lidx, nf, sf)


class EulerBCInters(BaseAdvecBCInters):
    _get_bc = get_bc

    def construct_kernels(self, elemap, impl_op):
        super().construct_kernels(elemap)
        
        # Collect face point array
        fpts = self._fpts

        if impl_op == 'spectral-radius':
            self.compute_flux = Kernel(
                *self._make_flux(impl_op), fpts, self.rank_fspr
            )
        elif impl_op == 'approx-jacobian':
            self.compute_flux = Kernel(
                *self._make_flux(impl_op), fpts, self.rank_jmat
            )
        else:
            self.compute_flux = Kernel(*self._make_flux(impl_op), fpts)

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

        # Get bc function (`self.bc` was defined at `baseadvec.inters`)
        bc = self.bc

        if impl_op == 'spectral-radius':
            # Get wave speed function
            wave_speed = self.ele0.make_wave_speed()

            def bc_flux_spr(i_begin, i_end, lidx, nf, sf, face_ids,
                            uf, lam):
                for idx in range(i_begin, i_end):
                    fn = array((nfvars,), np.float64)
                    ur = array((nfvars,), np.float64)

                    # Normal vector
                    nfi = nf[:, idx]

                    # Left solutions
                    lti, lei, lfi = lidx[:, idx]
                    ul = uf[lti][lfi, :, lei]

                    # Compute BC
                    bc(ul, ur, nfi)

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)
                    
                    # Compute spectral radius on face
                    lami = wave_speed(ul, nfi)
                    lam[face_ids[idx]] = lami

                    for jdx in range(nfvars):
                        # Save it at left solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

            return self.be.make_loop(
                self.nfpts, bc_flux_spr, lidx, nf, sf,
                self.rank_face_ids
            )
        elif impl_op == 'approx-jacobian':
            from pybaram.solvers.euler.jacobian import make_convective_jacobian

            # Get Jacobian functions
            pos_jacobian = make_convective_jacobian(self.be, cplargs, 'positive')

            def bc_flux_ajac(i_begin, i_end, lidx, nf, sf, face_ids,
                             uf, jmat):
                for idx in range(i_begin, i_end):
                    fn = array((nfvars,), np.float64)
                    ur = array((nfvars,), np.float64)
                    ap = array((nfvars, nfvars), np.float64)

                    # Normal vector
                    nfi = nf[:, idx]

                    # Left solutions
                    lti, lei, lfi = lidx[:, idx]
                    ul = uf[lti][lfi, :, lei]

                    # Compute BC
                    bc(ul, ur, nfi)

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)
                    
                    # Compute Jacobian matrix on face
                    pos_jacobian(ul, nfi, ap)
                    for row in range(nfvars):
                        for col in range(nfvars):
                            jmat[0, row, col, face_ids[idx]] = ap[row][col]

                    for jdx in range(nfvars):
                        # Save it at left solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

            return self.be.make_loop(
                self.nfpts, bc_flux_ajac, lidx, nf, sf,
                self.rank_face_ids
            )
        else:
            def bc_flux(i_begin, i_end, lidx, nf, sf, uf):
                for idx in range(i_begin, i_end):
                    fn = array((nfvars,), np.float64)
                    ur = array((nfvars,), np.float64)

                    # Normal vector
                    nfi = nf[:, idx]

                    # Left solutions
                    lti, lei, lfi = lidx[:, idx]
                    ul = uf[lti][lfi, :, lei]

                    # Compute BC
                    bc(ul, ur, nfi)

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)

                    for jdx in range(nfvars):
                        # Save it at left solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

        return self.be.make_loop(self.nfpts, bc_flux, lidx, nf, sf)


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
