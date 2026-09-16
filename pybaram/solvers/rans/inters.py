# -*- coding: utf-8 -*-
from pybaram.solvers.baseadvecdiff import BaseAdvecDiffIntInters, BaseAdvecDiffBCInters, BaseAdvecDiffMPIInters
from pybaram.backends.types import Kernel
from pybaram.solvers.rans.visflux import make_visflux
from pybaram.solvers.euler.rsolvers import get_rsolver
from pybaram.utils.np import npeval

import numpy as np
import re


class RANSIntInters(BaseAdvecDiffIntInters):
    def construct_kernels(self, elemap, impl_op):
        # Wall distance at face
        ydistf = [cell.rawydist for cell in elemap.values()]
        _ydist = np.array([ydistf[t][e]  for (t, e, _) in self.rawlidx.T])
        self.ydist = self.be.convert_array(_ydist)

        # Call Parent method
        super().construct_kernels(elemap)

        # Save viscosity on face (for implicit operator)
        self.muf = muf = self.be.alloc_array((2, self.nfpts))

        # Collect face point array
        fpts, gradf = self._fpts, self._gradf

        if impl_op == 'spectral-radius':
            self.compute_flux = Kernel(
                *self._make_flux(impl_op), muf, gradf, fpts,
                self.rank_fspr, self.rank_tfspr
            )
        elif impl_op == 'approx-jacobian':
            self.compute_flux = Kernel(
                *self._make_flux(impl_op), muf, gradf, fpts,
                self.rank_jmat, self.rank_tjmat
            )
        else:
            self.compute_flux = Kernel(*self._make_flux(impl_op), muf, gradf, fpts)

    def _make_flux(self, impl_op):
        ndims, nvars, nfvars = self.ndims, self.nvars, self.nfvars

        # Constant arrays
        lidx = self.lidx
        ridx = self.ridx
        nf, sf = self.vec_snorm, self.mag_snorm
        is_axi = getattr(self.ele0, '_is_axisymmetric', False)
        rf = self.axisymmetric_radius if is_axi else self.mag_snorm
        ydist = self.ydist
        rcp_dx = self.rcp_dx

        # Compiler arguments
        array = self.be.local()
        cplargs = {
            'flux' : self.ele0.flux_container(),
            'to_primevars' : self.ele0.to_flow_primevars(),
            'ndims' : ndims,
            'nfvars' : nfvars,
            'array' : array,
            'axisymmetric' : is_axi,
            'axisymmetric_radius_idx' : getattr(self.ele0, '_axisymmetric_radius_idx', 1),
            **self._const
        }

        # Get numerical schems from `rsolvers.py`
        scheme = self.cfg.get('solver', 'riemann-solver')
        flux = get_rsolver(scheme, self.be, cplargs)

        # Get compiled function of viscosity and viscous flux
        compute_mu = self.ele0.mu_container()
        compute_mut = self.ele0.mut_container()
        visflux = make_visflux(self.be, cplargs)

        # Get turbulence flux from `turbulent.py`
        tflux = self._make_turb_flux()

        if impl_op == 'spectral-radius':
            # Get wave speed function
            wave_speed = self.ele0.make_wave_speed()
            twave_speed = self.ele0.make_turb_wave_speed()

            def comm_flux_spr(i_begin, i_end, lidx, ridx, nf, sf, rf,
                              rcp_dx, ydist, face_ids, muf, gradf, uf,
                              lam, tlam):
                for idx in range(i_begin, i_end):
                    fn = array((nvars,), np.float64)
                    um = array((nvars,), np.float64)

                    # Normal vector and wall distance (ydns)
                    nfi = nf[:, idx]
                    ydnsi = ydist[idx]
                    rcp_dxi = rcp_dx[idx]

                    # Left and right solutions
                    lti, lei, lfi = lidx[:, idx]
                    rti, rei, rfi = ridx[:, idx]
                    ul = uf[lti][lfi, :, lei]
                    ur = uf[rti][rfi, :, rei]

                    # Gradient and solution at face
                    gf = gradf[:, :, idx]

                    for jdx in range(nvars):
                        um[jdx] = 0.5*(ul[jdx] + ur[jdx])

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)
                    
                    # Compute viscosity and viscous flux
                    muf[0, idx] = mu = compute_mu(um)
                    muf[1, idx] = mut = compute_mut(um, gf, mu, ydnsi)
                    if is_axi:
                        visflux(um, gf, nfi, mu, mut, rf[idx], fn)
                    else:
                        visflux(um, gf, nfi, mu, mut, fn)

                    # Compute turbulent flux
                    tflux(ul, ur, um, gf, nfi, ydnsi, mu, mut, fn)

                    # Compute wave speed on both cell
                    laml = wave_speed(ul, nfi, rcp_dxi, mu, mut)
                    lamr = wave_speed(ur, nfi, rcp_dxi, mu, mut)

                    # Compute spectral radius on face
                    lami = max(laml, lamr)
                    lam[face_ids[idx]] = lami

                    # Compute turbulent spectral radius
                    tlami = twave_speed(um, nfi, rcp_dxi, mu, mut)
                    tlam[face_ids[idx]] = tlami

                    for jdx in range(nvars):
                        # Save it at left and right solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]
                        uf[rti][rfi, jdx, rei] = -fn[jdx]*sf[idx]

            return self.be.make_loop(
                self.nfpts, comm_flux_spr, lidx, ridx, nf, sf, rf,
                rcp_dx, ydist, self.rank_face_ids
            )
        elif impl_op == 'approx-jacobian':
            from pybaram.solvers.euler.jacobian import make_convective_jacobian
            from pybaram.solvers.navierstokes.jacobian import get_viscous_jacobian

            ntvars = nvars - nfvars
            vistype = self.cfg.get('solver-time-integrator', 'visflux-jacobian', 'tlns')

            # Get Jacobian functions
            pos_conv_jac = make_convective_jacobian(self.be, cplargs, 'positive')
            neg_conv_jac = make_convective_jacobian(self.be, cplargs, 'negative')
            pos_visc_jac = get_viscous_jacobian(vistype, self.be, cplargs, 'positive')
            neg_visc_jac = get_viscous_jacobian(vistype, self.be, cplargs, 'negative')
            pos_turb_jac = self.ele0.make_turb_jacobian('positive')
            neg_turb_jac = self.ele0.make_turb_jacobian('negative')

            def comm_flux_ajac(i_begin, i_end, lidx, ridx, nf, sf, rf,
                               rcp_dx, ydist, face_ids, muf, gradf, uf,
                               jmat, tjmat):
                for idx in range(i_begin, i_end):
                    fn = array((nvars,), np.float64)
                    um = array((nvars,), np.float64)

                    ap = array((nfvars, nfvars), np.float64)
                    am = array((nfvars, nfvars), np.float64)
                    tap = array((ntvars, ntvars), np.float64)
                    tam = array((ntvars, ntvars), np.float64)

                    # Normal vector and wall distance (ydns)
                    nfi = nf[:, idx]
                    ydnsi = ydist[idx]
                    rcp_dxi = rcp_dx[idx]

                    # Left and right solutions
                    lti, lei, lfi = lidx[:, idx]
                    rti, rei, rfi = ridx[:, idx]
                    ul = uf[lti][lfi, :, lei]
                    ur = uf[rti][rfi, :, rei]

                    # Gradient and solution at face
                    gf = gradf[:, :, idx]

                    for jdx in range(nvars):
                        um[jdx] = 0.5*(ul[jdx] + ur[jdx])

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)
                    
                    # Compute viscosity and viscous flux
                    muf[0, idx] = mu = compute_mu(um)
                    muf[1, idx] = mut = compute_mut(um, gf, mu, ydnsi)
                    if is_axi:
                        visflux(um, gf, nfi, mu, mut, rf[idx], fn)
                    else:
                        visflux(um, gf, nfi, mu, mut, fn)

                    # Compute turbulent flux
                    tflux(ul, ur, um, gf, nfi, ydnsi, mu, mut, fn)

                    # Compute Jacobian matrix on surface
                    # based on left/right cell
                    pos_conv_jac(ul, nfi, ap)
                    neg_conv_jac(ur, nfi, am)

                    pos_visc_jac(ul, nfi, ap, mu, mut, rcp_dxi)
                    neg_visc_jac(ur, nfi, am, mu, mut, rcp_dxi)

                    for row in range(nfvars):
                        for col in range(nfvars):
                            face = face_ids[idx]
                            jmat[0, row, col, face] = ap[row][col]
                            jmat[1, row, col, face] = am[row][col]

                    pos_turb_jac(um, nfi, tap, rcp_dxi, mu, mut, gf, ydnsi)
                    neg_turb_jac(um, nfi, tam, rcp_dxi, mu, mut, gf, ydnsi)

                    for row in range(ntvars):
                        for col in range(ntvars):
                            face = face_ids[idx]
                            tjmat[0, row, col, face] = tap[row][col]
                            tjmat[1, row, col, face] = tam[row][col]

                    for jdx in range(nvars):
                        # Save it at left and right solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]
                        uf[rti][rfi, jdx, rei] = -fn[jdx]*sf[idx]

            return self.be.make_loop(
                self.nfpts, comm_flux_ajac, lidx, ridx, nf, sf, rf,
                rcp_dx, ydist, self.rank_face_ids
            )
        else:
            def comm_flux(i_begin, i_end, lidx, ridx, nf, sf, rf, ydist, muf, gradf, uf):
                for idx in range(i_begin, i_end):
                    fn = array((nvars,), np.float64)
                    um = array((nvars,), np.float64)

                    # Normal vector and wall distance (ydns)
                    nfi = nf[:, idx]
                    ydnsi = ydist[idx]

                    # Left and right solutions
                    lti, lei, lfi = lidx[:, idx]
                    rti, rei, rfi = ridx[:, idx]
                    ul = uf[lti][lfi, :, lei]
                    ur = uf[rti][rfi, :, rei]

                    # Gradient and solution at face
                    gf = gradf[:, :, idx]

                    for jdx in range(nvars):
                        um[jdx] = 0.5*(ul[jdx] + ur[jdx])

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)
                    
                    # Compute viscosity and viscous flux
                    muf[0, idx] = mu = compute_mu(um)
                    muf[1, idx] = mut = compute_mut(um, gf, mu, ydnsi)
                    if is_axi:
                        visflux(um, gf, nfi, mu, mut, rf[idx], fn)
                    else:
                        visflux(um, gf, nfi, mu, mut, fn)

                    # Compute turbulent flux
                    tflux(ul, ur, um, gf, nfi, ydnsi, mu, mut, fn)

                    for jdx in range(nvars):
                        # Save it at left and right solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]
                        uf[rti][rfi, jdx, rei] = -fn[jdx]*sf[idx]

            return self.be.make_loop(self.nfpts, comm_flux, lidx, ridx, nf, sf, rf, ydist)


class RANSMPIInters(BaseAdvecDiffMPIInters):
    def construct_kernels(self, elemap, impl_op):
        # Wall distance at face
        ydistf = [cell.rawydist for cell in elemap.values()]
        _ydist = np.array([ydistf[t][e]  for (t, e, _) in self.rawlidx.T])
        self.ydist = self.be.convert_array(_ydist)

        # Call Parent method
        super().construct_kernels(elemap)

        # Save viscosity on face (for implicit operator)
        self.muf = muf = self.be.alloc_array((2, self.nfpts))

        # Kernel to compute flux
        fpts, gradf = self._fpts, self._gradf
        rhs = self._rhs

        if impl_op == 'spectral-radius':
            self.compute_flux = Kernel(
                *self._make_flux(impl_op), muf, gradf, rhs, fpts,
                self.rank_fspr, self.rank_tfspr
            )
        elif impl_op == 'approx-jacobian':
            self.compute_flux = Kernel(
                *self._make_flux(impl_op), muf, gradf, rhs, fpts,
                self.rank_jmat, self.rank_tjmat
            )
        else:
            self.compute_flux = Kernel(*self._make_flux(impl_op), muf, gradf, rhs, fpts)

    def _make_flux(self, impl_op):
        ndims, nvars, nfvars = self.ndims, self.nvars, self.nfvars

        lidx = self.lidx
        nf, sf = self.vec_snorm, self.mag_snorm
        is_axi = getattr(self.ele0, '_is_axisymmetric', False)
        rf = self.axisymmetric_radius if is_axi else self.mag_snorm
        ydist = self.ydist
        rcp_dx = self.rcp_dx

        # Compiler arguments
        array = self.be.local()
        cplargs = {
            'flux' : self.ele0.flux_container(),
            'to_primevars' : self.ele0.to_flow_primevars(),
            'ndims' : ndims,
            'nfvars' : nfvars,
            'array' : array,
            'axisymmetric' : is_axi,
            'axisymmetric_radius_idx' : getattr(self.ele0, '_axisymmetric_radius_idx', 1),
            **self._const
        }

        # Get numerical schems from `rsolvers.py`
        scheme = self.cfg.get('solver', 'riemann-solver')
        flux = get_rsolver(scheme, self.be, cplargs)

        # Get compiled function of viscosity and viscous flux
        compute_mu = self.ele0.mu_container()
        compute_mut = self.ele0.mut_container()
        visflux = make_visflux(self.be, cplargs)

        # Get turbulence flux from `turbulent.py`
        tflux = self._make_turb_flux()
            
        if impl_op == 'spectral-radius':
            # Get wave speed function
            wave_speed = self.ele0.make_wave_speed()
            twave_speed = self.ele0.make_turb_wave_speed()

            def comm_flux_spr(i_begin, i_end, lidx, nf, sf, rf, rcp_dx,
                              ydist, face_ids, muf, gradf, rhs, uf, lam,
                              tlam):
                for idx in range(i_begin, i_end):
                    fn = array((nvars,), np.float64)
                    um = array((nvars,), np.float64)

                    # Normal vector and wall distance (ydns)
                    nfi = nf[:, idx]
                    ydnsi = ydist[idx]
                    rcp_dxi = rcp_dx[idx]

                    # Left and right solutions
                    lti, lei, lfi = lidx[:, idx]
                    ul = uf[lti][lfi, :, lei]
                    ur = rhs[:, idx]

                    # Gradient and solution at face
                    gf = gradf[:, :, idx]

                    for jdx in range(nvars):
                        um[jdx] = 0.5*(ul[jdx] + ur[jdx])

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)
                    
                    # Compute viscosity and viscous flux
                    muf[0, idx] = mu = compute_mu(um)
                    muf[1, idx] = mut = compute_mut(um, gf, mu, ydnsi)
                    if is_axi:
                        visflux(um, gf, nfi, mu, mut, rf[idx], fn)
                    else:
                        visflux(um, gf, nfi, mu, mut, fn)

                    # Compute turbulent flux
                    tflux(ul, ur, um, gf, nfi, ydnsi, mu, mut, fn)

                    # Compute spectral radius on face
                    lami = wave_speed(ul, nfi, rcp_dxi, mu, mut)
                    lam[face_ids[idx]] = lami

                    # Compute turbulent spectral radius
                    tlami = twave_speed(ul, nfi, rcp_dxi, mu, mut)
                    tlam[face_ids[idx]] = tlami

                    for jdx in range(nvars):
                        # Save it at left solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

            return self.be.make_loop(
                self.nfpts, comm_flux_spr, lidx, nf, sf, rf, rcp_dx,
                ydist, self.rank_face_ids
            )
        elif impl_op == 'approx-jacobian':
            from pybaram.solvers.euler.jacobian import make_convective_jacobian
            from pybaram.solvers.navierstokes.jacobian import get_viscous_jacobian

            ntvars = nvars - nfvars

            vistype = self.cfg.get('solver-time-integrator', 'visflux-jacobian', 'tlns')

            # Get Jacobian functions
            com_conv_jac = make_convective_jacobian(self.be, cplargs, 'positive')
            com_visc_jac = get_viscous_jacobian(vistype, self.be, cplargs)
            com_turb_jac = self.ele0.make_turb_jacobian()

            def comm_flux_ajac(i_begin, i_end, lidx, nf, sf, rf, rcp_dx,
                               ydist, face_ids, muf, gradf, rhs, uf,
                               jmat, tjmat):
                for idx in range(i_begin, i_end):
                    fn = array((nvars,), np.float64)
                    um = array((nvars,), np.float64)

                    ap = array((nfvars, nfvars), np.float64)
                    at = array((ntvars, ntvars), np.float64)

                    # Normal vector and wall distance (ydns)
                    nfi = nf[:, idx]
                    ydnsi = ydist[idx]
                    rcp_dxi = rcp_dx[idx]

                    # Left and right solutions
                    lti, lei, lfi = lidx[:, idx]
                    ul = uf[lti][lfi, :, lei]
                    ur = rhs[:, idx]

                    # Gradient and solution at face
                    gf = gradf[:, :, idx]

                    for jdx in range(nvars):
                        um[jdx] = 0.5*(ul[jdx] + ur[jdx])

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)
                    
                    # Compute viscosity and viscous flux
                    muf[0, idx] = mu = compute_mu(um)
                    muf[1, idx] = mut = compute_mut(um, gf, mu, ydnsi)
                    if is_axi:
                        visflux(um, gf, nfi, mu, mut, rf[idx], fn)
                    else:
                        visflux(um, gf, nfi, mu, mut, fn)

                    # Compute turbulent flux
                    tflux(ul, ur, um, gf, nfi, ydnsi, mu, mut, fn)

                    # Compute Jacobian matrix on surface
                    # based on left/right cell
                    com_conv_jac(ul, nfi, ap)
                    com_visc_jac(ul, nfi, ap, mu, mut, rcp_dxi)

                    for row in range(nfvars):
                        for col in range(nfvars):
                            jmat[0, row, col, face_ids[idx]] = ap[row][col]

                    # Turbulent Jacobian
                    com_turb_jac(ul, nfi, at, rcp_dxi, mu, mut, gf, ydnsi)
                    for row in range(ntvars):
                        for col in range(ntvars):
                            tjmat[0, row, col, face_ids[idx]] = at[row][col]

                    for jdx in range(nvars):
                        # Save it at left solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

            return self.be.make_loop(
                self.nfpts, comm_flux_ajac, lidx, nf, sf, rf, rcp_dx,
                ydist, self.rank_face_ids
            )
        else:
            def comm_flux(i_begin, i_end, lidx, nf, sf, rf, ydist, muf, gradf, rhs, uf):
                for idx in range(i_begin, i_end):
                    fn = array((nvars,), np.float64)
                    um = array((nvars,), np.float64)

                    # Normal vector and wall distance (ydns)
                    nfi = nf[:, idx]
                    ydnsi = ydist[idx]

                    # Left and right solutions
                    lti, lei, lfi = lidx[:, idx]
                    ul = uf[lti][lfi, :, lei]
                    ur = rhs[:, idx]

                    # Gradient and solution at face
                    gf = gradf[:, :, idx]

                    for jdx in range(nvars):
                        um[jdx] = 0.5*(ul[jdx] + ur[jdx])

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)
                    
                    # Compute viscosity and viscous flux
                    muf[0, idx] = mu = compute_mu(um)
                    muf[1, idx] = mut = compute_mut(um, gf, mu, ydnsi)
                    if is_axi:
                        visflux(um, gf, nfi, mu, mut, rf[idx], fn)
                    else:
                        visflux(um, gf, nfi, mu, mut, fn)

                    # Compute turbulent flux
                    tflux(ul, ur, um, gf, nfi, ydnsi, mu, mut, fn)

                    for jdx in range(nvars):
                        # Save it at left solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

            return self.be.make_loop(self.nfpts, comm_flux, lidx, nf, sf, rf, ydist)


class RANSBCInters(BaseAdvecDiffBCInters):
    is_vis_wall = False

    def construct_bc(self):
        # Parse BC function name
        bcf = re.sub('-', '_', self.name)

        # Constants for BC function
        if self._reqs:
            bcsect = 'soln-bcs-{}'.format(self.bctype)
            bcc = {k: npeval(self.cfg.getexpr(bcsect, k, self._const))
                   for k in self._reqs}
        else:
            bcc = {}

        bcc['ndims'], bcc['nvars'], bcc['nfvars'] = self.ndims, self.nvars, self.nfvars

        bcc.update(self._const)
        bcc.update(self._turb_coeffs)

        # Get bc from `bcs.py` (in rans...) and compile them
        self.bc = self._get_bc(self.be, bcf, bcc)

    def construct_kernels(self, elemap, impl_op):
        # Wall distance at face
        ydistf = [cell.rawydist for cell in elemap.values()]
        _ydist = np.array([ydistf[t][e]  for (t, e, _) in self.rawlidx.T])
        self.ydist = self.be.convert_array(_ydist)

        # Call Parent method
        super().construct_kernels(elemap)

        # Save viscosity on face (for implicit operator)
        self.muf = muf = self.be.alloc_array((2,self.nfpts))

        # Kernel to compute flux
        fpts, gradf = self._fpts, self._gradf

        if impl_op == 'spectral-radius':
            self.compute_flux = Kernel(
                *self._make_flux(impl_op), muf, gradf, fpts,
                self.rank_fspr, self.rank_tfspr
            )
        elif impl_op == 'approx-jacobian':
            self.compute_flux = Kernel(
                *self._make_flux(impl_op), muf, gradf, fpts,
                self.rank_jmat, self.rank_tjmat
            )
        else:
            self.compute_flux = Kernel(*self._make_flux(impl_op), muf, gradf, fpts)

    def _make_delu(self):
        nvars, ndims = self.nvars, self.ndims
        lidx = self.lidx
        nf = self.vec_snorm
        ydist = self.ydist

        # Compile functions
        array = self.be.local()
        compute_mu = self.ele0.mu_container()

        bc = self.bc

        def compute_delu(i_begin, i_end, lidx, nf, ydist, uf):
            for idx in range(i_begin, i_end):
                ur = array((nvars,), np.float64)
                nfi = nf[:, idx]

                lti, lei, lfi = lidx[:, idx]

                ul = uf[lti][lfi, :, lei]
                
                mul = compute_mu(ul)
                bc(ul, ur, nfi, mul, ydist[idx])

                for jdx in range(nvars):
                    du = ur[jdx] - ul[jdx]
                    uf[lti][lfi, jdx, lei] = du

        return self.be.make_loop(self.nfpts, compute_delu, lidx, nf, ydist)
        
    def _make_flux(self, impl_op):
        ndims, nvars, nfvars = self.ndims, self.nvars, self.nfvars

        lidx = self.lidx
        nf, sf = self.vec_snorm, self.mag_snorm
        is_axi = getattr(self.ele0, '_is_axisymmetric', False)
        rf = self.axisymmetric_radius if is_axi else self.mag_snorm
        ydist = self.ydist
        rcp_dx = self.rcp_dx

        # Compiler arguments
        array = self.be.local()
        cplargs = {
            'flux' : self.ele0.flux_container(),
            'to_primevars' : self.ele0.to_flow_primevars(),
            'ndims' : ndims,
            'nfvars' : nfvars,
            'array' : array,
            'axisymmetric' : is_axi,
            'axisymmetric_radius_idx' : getattr(self.ele0, '_axisymmetric_radius_idx', 1),
            **self._const
        }

        # Get numerical schems from `rsolvers.py`
        scheme = self.cfg.get('solver', 'riemann-solver')
        flux = get_rsolver(scheme, self.be, cplargs)

        # Get compiled function of viscosity and viscous flux
        compute_mu = self.ele0.mu_container()
        compute_mut = self.ele0.mut_container()
        visflux = make_visflux(self.be, cplargs)

        # Get turbulence flux from `turbulent.py`
        tflux = self._make_turb_flux()

        # Get bc function (`self.bc` was defined at `baseadvec.inters`)
        bc = self.bc

        if impl_op == 'spectral-radius':
            # Get wave speed function
            wave_speed = self.ele0.make_wave_speed()
            twave_speed = self.ele0.make_turb_wave_speed()

            def comm_flux_spr(i_begin, i_end, lidx, nf, sf, rf, rcp_dx,
                              ydist, face_ids, muf, gradf, uf, lam, tlam):
                for idx in range(i_begin, i_end):
                    fn = array((nvars,), np.float64)
                    um = array((nvars,), np.float64)
                    ur = array((nvars,), np.float64)

                    # Normal vector and wall distance (ydns)
                    nfi = nf[:, idx]
                    ydnsi = ydist[idx]
                    rcp_dxi = rcp_dx[idx]

                    # Left solutions
                    lti, lei, lfi = lidx[:, idx]
                    ul = uf[lti][lfi, :, lei]

                    # Gradient at face
                    gf = gradf[:, :, idx]

                    # Viscosity from left solution
                    mul = compute_mu(ul)

                    # Compute BC
                    bc(ul, ur, nfi, mul, ydnsi)

                    # Solution at face
                    for jdx in range(nvars):
                        um[jdx] = 0.5*(ul[jdx] + ur[jdx])

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)
                    
                    # Compute viscosity and viscous flux
                    muf[0, idx] = mu = compute_mu(um)
                    muf[1, idx] = mut = compute_mut(um, gf, mu, ydnsi)
                    if is_axi:
                        visflux(um, gf, nfi, mu, mut, rf[idx], fn)
                    else:
                        visflux(um, gf, nfi, mu, mut, fn)

                    # Compute turbulent flux
                    tflux(ul, ur, um, gf, nfi, ydnsi, mu, mut, fn)

                    # Compute spectral radius on face
                    lami = wave_speed(ul, nfi, rcp_dxi, mu, mut)
                    lam[face_ids[idx]] = lami

                    # Compute turbulent spectral radius
                    tlami = twave_speed(ul, nfi, rcp_dxi, mu, mut)
                    tlam[face_ids[idx]] = tlami

                    for jdx in range(nvars):
                        # Save it at left solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

            return self.be.make_loop(
                self.nfpts, comm_flux_spr, lidx, nf, sf, rf, rcp_dx,
                ydist, self.rank_face_ids
            )
        elif impl_op == 'approx-jacobian':
            from pybaram.solvers.euler.jacobian import make_convective_jacobian
            from pybaram.solvers.navierstokes.jacobian import get_viscous_jacobian

            ntvars = nvars - nfvars

            vistype = self.cfg.get('solver-time-integrator', 'visflux-jacobian', 'tlns')

            # Get Jacobian functions
            com_conv_jac = make_convective_jacobian(self.be, cplargs, 'positive')
            com_visc_jac = get_viscous_jacobian(vistype, self.be, cplargs)
            com_turb_jac = self.ele0.make_turb_jacobian()

            def comm_flux_ajac(i_begin, i_end, lidx, nf, sf, rf, rcp_dx,
                               ydist, face_ids, muf, gradf, uf, jmat,
                               tjmat):
                for idx in range(i_begin, i_end):
                    fn = array((nvars,), np.float64)
                    um = array((nvars,), np.float64)
                    ur = array((nvars,), np.float64)

                    ap = array((nfvars, nfvars), np.float64)
                    at = array((ntvars, ntvars), np.float64)

                    # Normal vector and wall distance (ydns)
                    nfi = nf[:, idx]
                    ydnsi = ydist[idx]
                    rcp_dxi = rcp_dx[idx]

                    # Left solutions
                    lti, lei, lfi = lidx[:, idx]
                    ul = uf[lti][lfi, :, lei]

                    # Gradient at face
                    gf = gradf[:, :, idx]

                    # Viscosity from left solution
                    mul = compute_mu(ul)

                    # Compute BC
                    bc(ul, ur, nfi, mul, ydnsi)

                    # Solution at face
                    for jdx in range(nvars):
                        um[jdx] = 0.5*(ul[jdx] + ur[jdx])

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)
                    
                    # Compute viscosity and viscous flux
                    muf[0, idx] = mu = compute_mu(um)
                    muf[1, idx] = mut = compute_mut(um, gf, mu, ydnsi)
                    if is_axi:
                        visflux(um, gf, nfi, mu, mut, rf[idx], fn)
                    else:
                        visflux(um, gf, nfi, mu, mut, fn)

                    # Compute turbulent flux
                    tflux(ul, ur, um, gf, nfi, ydnsi, mu, mut, fn)

                    # Compute Jacobian matrix on surface
                    # based on left/right cell
                    com_conv_jac(ul, nfi, ap)
                    com_visc_jac(ul, nfi, ap, mu, mut, rcp_dxi)

                    for row in range(nfvars):
                        for col in range(nfvars):
                            jmat[0, row, col, face_ids[idx]] = ap[row][col]

                    com_turb_jac(ul, nfi, at, rcp_dxi, mu, mut, gf, ydnsi)
                    for row in range(ntvars):
                        for col in range(ntvars):
                            tjmat[0, row, col, face_ids[idx]] = at[row][col]

                    for jdx in range(nvars):
                        # Save it at left solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

            return self.be.make_loop(
                self.nfpts, comm_flux_ajac, lidx, nf, sf, rf, rcp_dx,
                ydist, self.rank_face_ids
            )
        else:
            def comm_flux(i_begin, i_end, lidx, nf, sf, rf, ydist, muf, gradf, uf):
                for idx in range(i_begin, i_end):
                    fn = array((nvars,), np.float64)
                    um = array((nvars,), np.float64)
                    ur = array((nvars,), np.float64)

                    # Normal vector and wall distance (ydns)
                    nfi = nf[:, idx]
                    ydnsi = ydist[idx]

                    # Left solutions
                    lti, lei, lfi = lidx[:, idx]
                    ul = uf[lti][lfi, :, lei]

                    # Gradient at face
                    gf = gradf[:, :, idx]

                    # Viscosity from left solution
                    mul = compute_mu(ul)

                    # Compute BC
                    bc(ul, ur, nfi, mul, ydnsi)

                    # Solution at face
                    for jdx in range(nvars):
                        um[jdx] = 0.5*(ul[jdx] + ur[jdx])

                    # Compute approixmate Riemann solver
                    flux(ul, ur, nfi, fn)
                    
                    # Compute viscosity and viscous flux
                    muf[0, idx] = mu = compute_mu(um)
                    muf[1, idx] = mut = compute_mut(um, gf, mu, ydnsi)
                    if is_axi:
                        visflux(um, gf, nfi, mu, mut, rf[idx], fn)
                    else:
                        visflux(um, gf, nfi, mu, mut, fn)

                    # Compute turbulent flux
                    tflux(ul, ur, um, gf, nfi, ydnsi, mu, mut, fn)

                    for jdx in range(nvars):
                        # Save it at left solution array
                        uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

            return self.be.make_loop(self.nfpts, comm_flux, lidx, nf, sf, rf, ydist)


class RANSSlipWallBCInters(RANSBCInters):
    name = 'slip-wall'


class RANSAdiaWallBCInters(RANSBCInters):
    name = 'adia-wall'
    is_vis_wall = True


class RANSIsothermWallBCInters(RANSBCInters):
    name = 'isotherm-wall'
    is_vis_wall = True
    _reqs = ['cptw']


class RANSSupOutBCInters(RANSBCInters):
    name = 'sup-out'


class RANSSupInBCInters(RANSBCInters):
    name = 'sup-in'

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)

        self._reqs = self.primevars


class RANSFarBCInters(RANSBCInters):
    name = 'far'

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)

        self._reqs = self.primevars


class RANSSubOutPBCInters(RANSBCInters):
    name = 'sub-outp'
    _reqs = ['p']


class RANSSubInvBCInters(RANSBCInters):
    name = 'sub-inv'

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)

        self._reqs = ['rho'] + ['u', 'v', 'w'][:self.ndims] + self.primevars[self.nfvars:]


class RANSSubInpttBCInters(RANSBCInters):
    name = 'sub-inptt'

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)

        self._reqs = ['p0', 'cpt0', 'dir'] + self.primevars[self.nfvars:]
