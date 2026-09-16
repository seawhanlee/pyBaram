# -*- coding: utf-8 -*-
from pybaram.solvers.baseadvec import BaseAdvecIntInters, BaseAdvecBCInters, BaseAdvecMPIInters
from pybaram.backends.types import (
    Kernel, NullKernel, MPIPackKernel, MPIUnpackKernel, MPISendKernel
)
from pybaram.utils.nb import dot

import numpy as np


class BaseAdvecDiffIntInters(BaseAdvecIntInters):
    def construct_kernels(self, elemap):
        # View of elemenet array (flux and gradient)
        self._fpts = fpts = tuple(cell.fpts for cell in elemap.values())
        dfpts = tuple(cell.grad for cell in elemap.values())

        # Array for gradient at face
        self._gradf = gradf = self.be.alloc_array((self.ndims, self.nvars, self.nfpts))

        # Allocate constant arrays
        self.rcp_dx = self.be.convert_array(self._rcp_dx)

        # Kernel to compute differnce of solution at face
        self.compute_delu = Kernel(*self._make_delu(), fpts)

        # Kernel to compute gradient at face (Averaging gradient)
        self.compute_grad_at = Kernel(
            *self._make_grad_at(), gradf, fpts, dfpts
        )

    def _make_grad_at(self):
        nvars, ndims = self.nvars, self.ndims
        lidx = self.lidx
        ridx = self.ridx

        # Mangitude and direction of the connecting vector
        inv_tf = self.rcp_dx
        _tf = self._dx_adj * self._rcp_dx
        tf = self.be.convert_array(_tf)
        avec = self.be.convert_array(self.raw_vec_snorm/np.einsum('ij,ij->j', _tf, self.raw_vec_snorm))

        # Stack-allocated array
        array = self.be.local()

        def grad_at(i_begin, i_end, lidx, ridx, inv_tf, tf, avec, gradf, du, gradu):
            for idx in range(i_begin, i_end):
                gf = array((ndims,), np.float64)

                lti, lei, lfi = lidx[:, idx]
                rti, rei, rfi = ridx[:, idx]

                tfi = tf[:, idx]
                inv_tfi = inv_tf[idx]
                aveci = avec[:, idx]

                # Compute the average of gradient at face
                for jdx in range(nvars):
                    gfl = gradu[lti][:, jdx, lei]
                    gfr = gradu[rti][:, jdx, rei]
                    for kdx in range(ndims):
                        gf[kdx] = 0.5*(gfl[kdx] + gfr[kdx])

                    gft = dot(gf, tfi, ndims, 0, 0)

                    # Compute gradient with jump term
                    for kdx in range(ndims):
                        gf[kdx] -= (gft - du[lti][lfi, jdx, lei]
                                    * inv_tfi)*aveci[kdx]
                        gradf[kdx, jdx, idx] = gf[kdx]

        return self.be.make_loop(self.nfpts, grad_at, lidx, ridx, inv_tf, tf, avec)


class BaseAdvecDiffMPIInters(BaseAdvecMPIInters):
    def construct_kernels(self, elemap):
        # Buffers
        self.rawlhs = self.be.alloc_array((self.nvars, self.nfpts), pinned=True)
        self.rawrhs = self.be.alloc_array((self.nvars, self.nfpts), pinned=True)

        self.lhs = lhs = self.be.convert_array(self.rawlhs)
        self._rhs = rhs = self.be.convert_array(self.rawrhs)

        # Gradient at face and buffer
        self.raw_gradf = self.be.alloc_array((self.ndims, self.nvars, self.nfpts), pinned=True)
        self.raw_grad_rhs = self.be.alloc_array((self.ndims, self.nvars, self.nfpts), pinned=True)
        self._gradf = gradf = self.be.convert_array(self.raw_gradf)
        self.grad_rhs = grad_rhs = self.be.convert_array(self.raw_grad_rhs)

        # View of element array
        self._fpts = fpts = tuple(cell.fpts for cell in elemap.values())
        dfpts = tuple(cell.grad for cell in elemap.values())

        # Allocate constant arrays
        self.rcp_dx = self.be.convert_array(self._rcp_dx)

        # Kernel to compute differnce of solution at face
        self.compute_delu = Kernel(*self._make_delu(), rhs, fpts)

        # Kernel to compute gradient at face (Averaging gradient)
        self.compute_grad_at = Kernel(
            *self._make_grad_at(), gradf, grad_rhs, fpts
        )

        # Kernel for pack, send, receive
        pack = Kernel(*self._make_pack(), lhs, fpts)
        send, self.sreq = self._make_send(self.rawlhs)
        self.send = MPISendKernel(self.be, send)
        self.recv, self.rreq = self._make_recv(self.rawrhs)

        pack_grad = Kernel(*self._make_pack_grad(), gradf, dfpts)
        send_grad, self.sgreq = self._make_send(self.raw_gradf)
        self.send_grad = MPISendKernel(self.be, send_grad)
        self.recv_grad, self.rgreq = self._make_recv(self.raw_grad_rhs)

        # Sync Host <-> Device
        if self.be.name == 'cuda':
            dtoh = Kernel(self.be.copy_array('d2h'), self.rawlhs, lhs)
            htod = Kernel(self.be.copy_array('h2d'), rhs, self.rawrhs)
            dtoh_grad = Kernel(self.be.copy_array('d2h'), self.raw_gradf, gradf)
            htod_grad = Kernel(self.be.copy_array('h2d'), grad_rhs, self.raw_grad_rhs)
        else:
            dtoh = NullKernel()
            htod = NullKernel()
            dtoh_grad = NullKernel()
            htod_grad = NullKernel()

        self.pack = MPIPackKernel(self.be, pack, dtoh)
        self.unpack = MPIUnpackKernel(self.be, htod)
        self.pack_grad = MPIPackKernel(self.be, pack_grad, dtoh_grad)
        self.unpack_grad = MPIUnpackKernel(self.be, htod_grad)

        self.pre_send = NullKernel()
        self.post_recv = self.unpack
        self.pre_send_grad = NullKernel()
        self.post_recv_grad = self.unpack_grad

    def _make_grad_at(self):
        nvars, ndims = self.nvars, self.ndims
        lidx = self.lidx

        # Mangitude and direction of the connecting vector
        inv_tf = self.rcp_dx
        _tf = self._dx_adj * self._rcp_dx
        tf = self.be.convert_array(_tf)
        avec = self.be.convert_array(self.raw_vec_snorm/np.einsum('ij,ij->j', _tf, self.raw_vec_snorm))

        # Stack-allocated array
        array = self.be.local()

        def grad_at(i_begin, i_end, lidx, inv_tf, tf, avec, gradf, grad_rhs, du):
            for idx in range(i_begin, i_end):
                gf = array((ndims,), np.float64)

                lti, lei, lfi = lidx[:, idx]

                tfi = tf[:, idx]
                inv_tfi = inv_tf[idx]
                aveci = avec[:, idx]

                # Compute the average of gradient at face
                for jdx in range(nvars):
                    for kdx in range(ndims):
                        gf[kdx] = 0.5*(gradf[kdx, jdx, idx] +
                                       grad_rhs[kdx, jdx, idx])

                    gft = dot(gf, tfi, ndims, 0, 0)

                    # Compute gradient with jump term
                    for kdx in range(ndims):
                        gf[kdx] -= (gft - du[lti][lfi, jdx, lei]
                                    * inv_tfi)*aveci[kdx]

                        gradf[kdx, jdx, idx] = gf[kdx]

        return self.be.make_loop(self.nfpts, grad_at, lidx, inv_tf, tf, avec)

    def _make_pack_grad(self):
        ndims, nvars = self.ndims, self.nvars
        lidx = self.lidx

        def pack(i_begin, i_end, lidx, lhs, uf):
            for idx in range(i_begin, i_end):
                lti, lei, lfi = lidx[:, idx]

                for jdx in range(nvars):
                    for kdx in range(ndims):
                        lhs[kdx, jdx, idx] = uf[lti][kdx, jdx, lei]

        return self.be.make_loop(self.nfpts, pack, lidx)


class BaseAdvecDiffBCInters(BaseAdvecBCInters):
    def construct_kernels(self, elemap):
        self.construct_bc()

        # View of elemenet array
        self._fpts = fpts = tuple(cell.fpts for cell in elemap.values())
        dfpts = tuple(cell.grad for cell in elemap.values())

        # Gradient at face
        self._gradf = gradf = self.be.alloc_array((self.ndims, self.nvars, self.nfpts))

        # Allocate constant array
        self.rcp_dx = self.be.convert_array(self._rcp_dx)

        # Kernel to compute differnce of solution at face
        self.compute_delu = Kernel(*self._make_delu(), fpts)

        # Kernel to compute gradient at face (Averaging gradient)
        self.compute_grad_at = Kernel(
            *self._make_grad_at(), gradf, fpts, dfpts
        )

    def _make_grad_at(self):
        nvars, ndims = self.nvars, self.ndims
        lidx = self.lidx

        # Mangitude and direction of the connecting vector
        inv_tf = self.rcp_dx
        _tf = self._dx_adj * self._rcp_dx
        tf = self.be.convert_array(_tf)
        avec = self.be.convert_array(self.raw_vec_snorm/np.einsum('ij,ij->j', _tf, self.raw_vec_snorm))

        # Stack-allocated array
        array = self.be.local()

        def grad_at(i_begin, i_end, lidx, inv_tf, tf, avec, gradf, du, gradu):
            for idx in range(i_begin, i_end):
                gf = array((ndims,), np.float64)

                lti, lei, lfi = lidx[:, idx]

                tfi = tf[:, idx]
                inv_tfi = inv_tf[idx]
                aveci = avec[:, idx]

                # Compute the average of gradient at face
                for jdx in range(nvars):
                    for kdx in range(ndims):
                        gf[kdx] = gradu[lti][kdx, jdx, lei]

                    gft = dot(gf, tfi, ndims, 0, 0)

                    # Compute gradient with jump term
                    for kdx in range(ndims):
                        gf[kdx] -= (gft - du[lti][lfi, jdx, lei]
                                    * inv_tfi)*aveci[kdx]
                        gradf[kdx, jdx, idx] = gf[kdx]

        return self.be.make_loop(self.nfpts, grad_at, lidx, inv_tf, tf, avec)
