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
        ydistf = [cell.ydist for cell in elemap.values()]
        self.ydist = np.array([ydistf[t][e]  for (t, e, _) in self._lidx.T])

        # Call Parent method
        super().construct_kernels(elemap)

        # Save viscosity on face (for implicit operator)
        muf = np.empty((2,self.nfpts))

        # Kernel to compute flux
        fpts, gradf = self._fpts, self._gradf
        self.compute_flux = Kernel(self._make_flux(), muf, gradf, *fpts)

        if impl_op == 'spectral-radius':
            # Kernel to compute Spectral radius
            nele = len(fpts)
            fspr = [cell.fspr for cell in elemap.values()]
            tfspr = [cell.tfspr for cell in elemap.values()]
            self.compute_spec_rad = Kernel(self._make_spec_rad(nele), muf, *fpts, *fspr, *tfspr)
        elif impl_op == 'approx-jacobian':
            # Kernel to compute Jacobian matrices
            nele = len(fpts)
            fjmat = [cell.jmat for cell in elemap.values()]
            tfjmat = [cell.tjmat for cell in elemap.values()]
            self.compute_aprx_jac = Kernel(self._make_aprx_jac(nele), muf, gradf, *fpts, *fjmat, *tfjmat)

    def _make_flux(self):
        ndims, nvars, nfvars = self.ndims, self.nvars, self.nfvars

        lt, le, lf = self._lidx
        rt, re, rf = self._ridx
        nf, sf = self._vec_snorm, self._mag_snorm
        ydist = self.ydist

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
        compute_mut = self.ele0.mut_container()
        visflux = make_visflux(self.be, cplargs)

        # Get turbulence flux from `turbulent.py`
        tflux = self._make_turb_flux()

        def comm_flux(i_begin, i_end, muf, gradf, *uf):
            for idx in range(i_begin, i_end):
                fn = array((nvars,))
                um = array((nvars,))

                # Normal vector and wall distance (ydns)
                nfi = nf[:, idx]
                ydnsi = ydist[idx]

                # Left and right solutions
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                rti, rfi, rei = rt[idx], rf[idx], re[idx]
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
                visflux(um, gf, nfi, mu, mut, fn)

                # Compute turbulent flux
                tflux(ul, ur, um, gf, nfi, ydnsi, mu, mut, fn)

                for jdx in range(nvars):
                    # Save it at left and right solution array
                    uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]
                    uf[rti][rfi, jdx, rei] = -fn[jdx]*sf[idx]

        return self.be.make_loop(self.nfpts, comm_flux)

    def _make_spec_rad(self, nele):
        nvars = self.nvars
        lt, le, lf = self._lidx
        rt, re, rf = self._ridx
        nf = self._vec_snorm
        
        # reciprocal of distance between two cells
        rcp_dx = self._rcp_dx

        # Get wave speed function
        array = self.be.local()
        wave_speed = self.ele0.make_wave_speed()
        twave_speed = self.ele0.make_turb_wave_speed()

        def comm_spr(i_begin, i_end, muf, *ufl):
            uf, _lam = ufl[:nele], ufl[nele:]
            lam, tlam = _lam[:nele], _lam[nele:]

            for idx in range(i_begin, i_end):
                um = array((nvars,))

                # Normal vector
                nfi = nf[:, idx]
                rcp_dxi = rcp_dx[idx]

                # Left and right solutions
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                rti, rfi, rei = rt[idx], rf[idx], re[idx]
                ul = uf[lti][lfi, :, lei]
                ur = uf[rti][rfi, :, rei]

                # Get viscosity on face (saved at rhside)
                mu = muf[0, idx]
                mut = muf[1, idx]

                # Compute wave speed on both cell
                laml = wave_speed(ul, nfi, rcp_dxi, mu, mut)
                lamr = wave_speed(ur, nfi, rcp_dxi, mu, mut)

                # Compute spectral radius on face
                lami = max(laml, lamr)
                lam[lti][lfi, lei] = lami
                lam[rti][rfi, rei] = lami

                for jdx in range(nvars):
                    um[jdx] = 0.5*(ul[jdx] + ur[jdx])

                # Compute turbulent spectral radius
                tlami = twave_speed(um, nfi, rcp_dxi, mu, mut)
                tlam[lti][lfi, lei] = tlami
                tlam[rti][rfi, rei] = tlami

        return self.be.make_loop(self.nfpts, comm_spr)
    
    def _make_aprx_jac(self, nele):
        from pybaram.solvers.euler.jacobian import make_convective_jacobian
        from pybaram.solvers.navierstokes.jacobian import get_viscous_jacobian

        nvars, nfvars = self.nvars, self.nfvars
        ntvars = nvars - nfvars

        # Left and right indices
        lt, le, lf = self._lidx
        rt, re, rf = self._ridx
        nf = self._vec_snorm
        ydist = self.ydist
        
        # reciprocal of distance between two cells
        rcp_dx = self._rcp_dx

        cplargs = {
            'ndims': self.ndims,
            'nfvars': self.nfvars,
            'gamma': self.ele0._const['gamma'],
            'pr': self.ele0._const['pr'],
            'to_prim': self.ele0.to_flow_primevars()
        }

        # Get viscous Jacobian type
        vistype = self.cfg.get('solver-time-integrator', 'visflux-jacobian', 'tlns')

        # Get Jacobian functions
        pos_jacobian = make_convective_jacobian(self.be, cplargs, 'positive')
        neg_jacobian = make_convective_jacobian(self.be, cplargs, 'negative')
        vis_pos_jacobian = get_viscous_jacobian(vistype, self.be, cplargs, 'positive')
        vis_neg_jacobian = get_viscous_jacobian(vistype, self.be, cplargs, 'negative')
        turb_pos_jacobian = self.ele0.make_turb_jacobian('positive')
        turb_neg_jacobian = self.ele0.make_turb_jacobian('negative')

        # Temporal array & matrix
        array = self.be.local()

        def comm_apj(i_begin, i_end, muf, gradf, *ufj):
            uf, _jmats = ufj[:nele], ufj[nele:]
            jmats, tjmats = _jmats[:nele], _jmats[nele:]

            for idx in range(i_begin, i_end):
                um = array((nvars,))
                ap = array((nfvars, nfvars))
                am = array((nfvars, nfvars))
                tap = array((ntvars, ntvars))
                tam = array((ntvars, ntvars))

                # Normal vector
                nfi = nf[:, idx]
                rcp_dxi = rcp_dx[idx]

                # Left and right solutions
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                rti, rfi, rei = rt[idx], rf[idx], re[idx]
                ul = uf[lti][lfi, :, lei]
                ur = uf[rti][rfi, :, rei]

                # Get viscosity on face (saved at rhside)
                mu = muf[0, idx]
                mut = muf[1, idx]

                # Compute Jacobian matrix on surface
                # based on left/right cell
                pos_jacobian(ul, nfi, ap)
                neg_jacobian(ur, nfi, am)

                vis_pos_jacobian(ul, nfi, ap, mu, rcp_dxi)
                vis_neg_jacobian(ur, nfi, am, mu, rcp_dxi)

                for row in range(nfvars):
                    for col in range(nfvars):
                        jmats[lti][0, row, col, lfi, lei] = ap[row][col]
                        jmats[lti][1, row, col, lfi, lei] = am[row][col]
                        jmats[rti][0, row, col, rfi, rei] = -am[row][col]
                        jmats[rti][1, row, col, rfi, rei] = -ap[row][col]
                
                # Turbulent Jacobian
                gf = gradf[:, :, idx]
                ydnsi = ydist[idx]
                for jdx in range(nvars):
                    um[jdx] = 0.5*(ul[jdx] + ur[jdx])
                
                turb_pos_jacobian(um, nfi, tap, rcp_dxi, mu, mut, gf, ydnsi)
                turb_neg_jacobian(um, nfi, tam, rcp_dxi, mu, mut, gf, ydnsi)

                for row in range(ntvars):
                    for col in range(ntvars):
                        tjmats[lti][0, row, col, lfi, lei] = tap[row][col]
                        tjmats[lti][1, row, col, lfi, lei] = tam[row][col]
                        tjmats[rti][0, row, col, rfi, rei] = -tam[row][col]
                        tjmats[rti][1, row, col, rfi, rei] = -tap[row][col]

        return self.be.make_loop(self.nfpts, comm_apj)


class RANSMPIInters(BaseAdvecDiffMPIInters):
    def construct_kernels(self, elemap, impl_op):
        # Wall distance at face
        ydistf = [cell.ydist for cell in elemap.values()]
        self.ydist = np.array([ydistf[t][e]  for (t, e, _) in self._lidx.T])

        # Call Parent method
        super().construct_kernels(elemap)

        # Save viscosity on face (for implicit operator)
        muf = np.empty((2,self.nfpts))

        # Kernel to compute flux
        fpts, gradf = self._fpts, self._gradf
        rhs = self._rhs
        self.compute_flux = Kernel(self._make_flux(), muf, gradf, rhs, *fpts)

        if impl_op == 'spectral-radius':
            # Kernel to compute Spectral radius
            nele = len(fpts)
            fspr = [cell.fspr for cell in elemap.values()]
            tfspr = [cell.tfspr for cell in elemap.values()]
            self.compute_spec_rad = Kernel(self._make_spec_rad(nele), muf, *fpts, *fspr, *tfspr)
        elif impl_op == 'approx-jacobian':
            # Kernel to compute Jacobian matrices
            nele = len(fpts)
            fjmat = [cell.jmat for cell in elemap.values()]
            tfjmat = [cell.tjmat for cell in elemap.values()]
            self.compute_aprx_jac = Kernel(self._make_aprx_jac(nele), muf, gradf, *fpts, *fjmat, *tfjmat)

    def _make_flux(self):
        ndims, nvars, nfvars = self.ndims, self.nvars, self.nfvars

        lt, le, lf = self._lidx
        nf, sf = self._vec_snorm, self._mag_snorm
        ydist = self.ydist

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
        compute_mut = self.ele0.mut_container()
        visflux = make_visflux(self.be, cplargs)

        # Get turbulence flux from `turbulent.py`
        tflux = self._make_turb_flux()

        def comm_flux(i_begin, i_end, muf, gradf, rhs, *uf):
            for idx in range(i_begin, i_end):
                fn = array((nvars,))
                um = array((nvars,))

                # Normal vector and wall distance (ydns)
                nfi = nf[:, idx]
                ydnsi = ydist[idx]

                # Left and right solutions
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
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
                visflux(um, gf, nfi, mu, mut, fn)

                # Compute turbulent flux
                tflux(ul, ur, um, gf, nfi, ydnsi, mu, mut, fn)

                for jdx in range(nvars):
                    # Save it at left solution array
                    uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

        return self.be.make_loop(self.nfpts, comm_flux)
    
    def _make_spec_rad(self, nele):
        lt, le, lf = self._lidx
        nf = self._vec_snorm
        
        # reciprocal of distance between two cells
        rcp_dx = self._rcp_dx

        # Get wave speed function
        wave_speed = self.ele0.make_wave_speed()
        twave_speed = self.ele0.make_turb_wave_speed()

        def comm_spr(i_begin, i_end, muf, *ufl):
            uf, _lam = ufl[:nele], ufl[nele:]
            lam, tlam = _lam[:nele], _lam[nele:]

            for idx in range(i_begin, i_end):
                # Normal vector
                nfi = nf[:, idx]
                rcp_dxi = rcp_dx[idx]

                # Left and right solutions
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                ul = uf[lti][lfi, :, lei]

                # Get viscosity on face (saved at rhside)
                mu = muf[0, idx]
                mut = muf[1, idx]

                # Compute spectral radius on face
                lami = wave_speed(ul, nfi, rcp_dxi, mu, mut)
                lam[lti][lfi, lei] = lami

                # Compute turbulent spectral radius
                tlami = twave_speed(ul, nfi, rcp_dxi, mu, mut)
                tlam[lti][lfi, lei] = tlami

        return self.be.make_loop(self.nfpts, comm_spr)
    
    def _make_aprx_jac(self, nele):
        from pybaram.solvers.euler.jacobian import make_convective_jacobian
        from pybaram.solvers.navierstokes.jacobian import get_viscous_jacobian

        nvars, nfvars = self.nvars, self.nfvars
        ntvars = nvars - nfvars

        # Left and right indices
        lt, le, lf = self._lidx
        nf = self._vec_snorm
        ydist = self.ydist
        
        # reciprocal of distance between two cells
        rcp_dx = self._rcp_dx

        cplargs = {
            'ndims': self.ndims,
            'nfvars': self.nfvars,
            'gamma': self.ele0._const['gamma'],
            'pr': self.ele0._const['pr'],
            'to_prim': self.ele0.to_flow_primevars()
        }

        # Get viscous Jacobian type
        vistype = self.cfg.get('solver-time-integrator', 'visflux-jacobian', 'tlns')

        # Get Jacobian functions
        pos_jacobian = make_convective_jacobian(self.be, cplargs, 'positive')
        vis_jacobian = get_viscous_jacobian(vistype, self.be, cplargs)
        turb_jacobian = self.ele0.make_turb_jacobian()

        # Temporal matrix
        array = self.be.local()

        def comm_apj(i_begin, i_end, muf, gradf, *ufj):
            uf, _jmats = ufj[:nele], ufj[nele:]
            jmats, tjmats = _jmats[:nele], _jmats[nele:]

            for idx in range(i_begin, i_end):
                ap = array((nfvars, nfvars))
                at = array((ntvars, ntvars))

                # Normal vector
                nfi = nf[:, idx]
                rcp_dxi = rcp_dx[idx]

                # Left and right solutions
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                ul = uf[lti][lfi, :, lei]

                # Get viscosity on face (saved at rhside)
                mu = muf[0, idx]
                mut = muf[1, idx]

                # Compute Jacobian matrix on surface
                # based on left/right cell
                pos_jacobian(ul, nfi, ap)
                vis_jacobian(ul, nfi, ap, mu, rcp_dxi)

                for row in range(nfvars):
                    for col in range(nfvars):
                        jmats[lti][0, row, col, lfi, lei] = ap[row][col]
                
                # Turbulent Jacobian
                gf = gradf[:, :, idx]
                ydnsi = ydist[idx]
                
                turb_jacobian(ul, nfi, at, rcp_dxi, mu, mut, gf, ydnsi)
                for row in range(ntvars):
                    for col in range(ntvars):
                        tjmats[lti][0, row, col, lfi, lei] = at[row][col]

        return self.be.make_loop(self.nfpts, comm_apj)


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
        ydistf = [cell.ydist for cell in elemap.values()]
        self.ydist = np.array([ydistf[t][e]  for (t, e, _) in self._lidx.T])

        # Call Parent method
        super().construct_kernels(elemap)

        # Save viscosity on face (for implicit operator)
        muf = np.empty((2,self.nfpts))

        # Kernel to compute flux
        fpts, gradf = self._fpts, self._gradf
        self.compute_flux = Kernel(self._make_flux(), muf, gradf, *fpts)

        if impl_op == 'spectral-radius':
            # Kernel to compute Spectral radius
            nele = len(fpts)
            fspr = [cell.fspr for cell in elemap.values()]
            tfspr = [cell.tfspr for cell in elemap.values()]
            self.compute_spec_rad = Kernel(self._make_spec_rad(nele), muf, *fpts, *fspr, *tfspr)
        elif impl_op == 'approx-jacobian':
            # Kernel to compute Jacobian matrices
            nele = len(fpts)
            fjmat = [cell.jmat for cell in elemap.values()]
            tfjmat = [cell.tjmat for cell in elemap.values()]
            self.compute_aprx_jac = Kernel(self._make_aprx_jac(nele), muf, gradf, *fpts, *fjmat, *tfjmat)

    def _make_delu(self):
        nvars, ndims = self.nvars, self.ndims
        lt, le, lf = self._lidx
        nf = self._vec_snorm
        ydist = self.ydist

        # Compile functions
        array = self.be.local()
        compute_mu = self.ele0.mu_container()

        bc = self.bc

        def compute_delu(i_begin, i_end, *uf):
            for idx in range(i_begin, i_end):
                ur = array((nvars,))
                nfi = nf[:, idx]

                lti, lfi, lei = lt[idx], lf[idx], le[idx]

                ul = uf[lti][lfi, :, lei]
                
                mul = compute_mu(ul)
                bc(ul, ur, nfi, mul, ydist[idx])

                for jdx in range(nvars):
                    du = ur[jdx] - ul[jdx]
                    uf[lti][lfi, jdx, lei] = du

        return self.be.make_loop(self.nfpts, compute_delu)
        
    def _make_flux(self):
        ndims, nvars, nfvars = self.ndims, self.nvars, self.nfvars

        lt, le, lf = self._lidx
        nf, sf = self._vec_snorm, self._mag_snorm
        ydist = self.ydist

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
        compute_mut = self.ele0.mut_container()
        visflux = make_visflux(self.be, cplargs)

        # Get turbulence flux from `turbulent.py`
        tflux = self._make_turb_flux()

        # Get bc function (`self.bc` was defined at `baseadvec.inters`)
        bc = self.bc

        def comm_flux(i_begin, i_end, muf, gradf, *uf):
            for idx in range(i_begin, i_end):
                fn = array((nvars,))
                um = array((nvars,))
                ur = array((nvars,))

                # Normal vector and wall distance (ydns)
                nfi = nf[:, idx]
                ydnsi = ydist[idx]

                # Left solutions
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
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
                visflux(um, gf, nfi, mu, mut, fn)

                # Compute turbulent flux
                tflux(ul, ur, um, gf, nfi, ydnsi, mu, mut, fn)

                for jdx in range(nvars):
                    # Save it at left solution array
                    uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

        return self.be.make_loop(self.nfpts, comm_flux)

    def _make_spec_rad(self, nele):
        lt, le, lf = self._lidx
        nf = self._vec_snorm
        
        # reciprocal of distance between two cells
        rcp_dx = self._rcp_dx

        # Get wave speed function
        wave_speed = self.ele0.make_wave_speed()
        twave_speed = self.ele0.make_turb_wave_speed()

        def comm_spr(i_begin, i_end, muf, *ufl):
            uf, _lam = ufl[:nele], ufl[nele:]
            lam, tlam = _lam[:nele], _lam[nele:]

            for idx in range(i_begin, i_end):
                # Normal vector
                nfi = nf[:, idx]
                rcp_dxi = rcp_dx[idx]

                # Left solution
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                ul = uf[lti][lfi, :, lei]

                # Get viscosity on face (saved at rhside)
                mu = muf[0, idx]
                mut = muf[1, idx]

                # Compute spectral radius on face
                lami = wave_speed(ul, nfi, rcp_dxi, mu, mut)
                lam[lti][lfi, lei] = lami

                # Compute turbulent spectral radius
                tlami = twave_speed(ul, nfi, rcp_dxi, mu, mut)
                tlam[lti][lfi, lei] = tlami

        return self.be.make_loop(self.nfpts, comm_spr)
    
    def _make_aprx_jac(self, nele):
        from pybaram.solvers.euler.jacobian import make_convective_jacobian
        from pybaram.solvers.navierstokes.jacobian import get_viscous_jacobian

        nvars, nfvars = self.nvars, self.nfvars
        ntvars = nvars - nfvars

        # Left and right indices
        lt, le, lf = self._lidx
        nf = self._vec_snorm
        ydist = self.ydist
        
        # reciprocal of distance between two cells
        rcp_dx = self._rcp_dx

        cplargs = {
            'ndims': self.ndims,
            'nfvars': self.nfvars,
            'gamma': self.ele0._const['gamma'],
            'pr': self.ele0._const['pr'],
            'to_prim': self.ele0.to_flow_primevars()
        }

        # Get viscous Jacobian type
        vistype = self.cfg.get('solver-time-integrator', 'visflux-jacobian', 'tlns')

        # Get Jacobian functions
        pos_jacobian = make_convective_jacobian(self.be, cplargs, 'positive')
        vis_jacobian = get_viscous_jacobian(vistype, self.be, cplargs)
        turb_jacobian = self.ele0.make_turb_jacobian()

        # Temporal matrix
        array = self.be.local()

        def comm_apj(i_begin, i_end, muf, gradf, *ufj):
            uf, _jmats = ufj[:nele], ufj[nele:]
            jmats, tjmats = _jmats[:nele], _jmats[nele:]

            for idx in range(i_begin, i_end):
                ap = array((nfvars, nfvars))
                at = array((ntvars, ntvars))

                # Normal vector
                nfi = nf[:, idx]
                rcp_dxi = rcp_dx[idx]

                # Left and right solutions
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                ul = uf[lti][lfi, :, lei]

                # Get viscosity on face (saved at rhside)
                mu = muf[0, idx]
                mut = muf[1, idx]

                # Compute Jacobian matrix on surface
                # based on left/right cell
                pos_jacobian(ul, nfi, ap)
                vis_jacobian(ul, nfi, ap, mu, rcp_dxi)

                for row in range(nfvars):
                    for col in range(nfvars):
                        jmats[lti][0, row, col, lfi, lei] = ap[row][col]
                
                # Turbulent Jacobian
                gf = gradf[:, :, idx]
                ydnsi = ydist[idx]
                
                turb_jacobian(ul, nfi, at, rcp_dxi, mu, mut, gf, ydnsi)
                for row in range(ntvars):
                    for col in range(ntvars):
                        tjmats[lti][0, row, col, lfi, lei] = at[row][col]

        return self.be.make_loop(self.nfpts, comm_apj)


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
