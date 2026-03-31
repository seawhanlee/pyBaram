# -*- coding: utf-8 -*-
from pybaram.solvers.base import BaseIntInters, BaseBCInters, BaseMPIInters
from pybaram.backends.types import Kernel, NullKernel
from pybaram.utils.np import npeval

import numpy as np
import re


class BaseAdvecIntInters(BaseIntInters):
    def construct_kernels(self, elemap):
        # View of elemenet array
        self._fpts = fpts = tuple(cell.fpts for cell in elemap.values())

        if self.order > 1:
            # Kernel to compute differnce of solution at face
            self.compute_delu = Kernel(*self._make_delu(), fpts)
        else:
            self.compute_delu = NullKernel

    def _make_delu(self):
        nvars = self.nvars

        def compute_delu(i_begin, i_end, lidx, ridx, uf):
            for idx in range(i_begin, i_end):
                lti, lei, lfi = lidx[:, idx]
                rti, rei, rfi = ridx[:, idx]

                for jdx in range(nvars):
                    ul = uf[lti][lfi, jdx, lei]
                    ur = uf[rti][rfi, jdx, rei]
                    du = ur - ul
                    uf[lti][lfi, jdx, lei] = du
                    uf[rti][rfi, jdx, rei] = -du

        return self.be.make_loop(self.nfpts, compute_delu, self.lidx, self.ridx)


class BaseAdvecMPIInters(BaseMPIInters):
    _tag = 1234

    def construct_kernels(self, elemap):
        # Buffers
        # TODO : Heterogeneous computing
        lhs = self.be.alloc_array((self.nvars, self.nfpts))
        self._rhs = rhs = self.be.alloc_array((self.nvars, self.nfpts))

        # View of elemenet array
        self._fpts = fpts = tuple(cell.fpts for cell in elemap.values())

        if self.order > 1:
            # Kernel to compute differnce of solution at face
            self.compute_delu = Kernel(*self._make_delu(), rhs, fpts)
        else:
            self.compute_delu = NullKernel

        # Kernel for pack, send, receive
        self.pack = Kernel(*self._make_pack(), lhs, fpts)
        self.send, self.sreq = self._make_send(lhs)
        self.recv, self.rreq = self._make_recv(rhs)

    def _make_delu(self):
        nvars = self.nvars
        lt, le, lf = self.rawlidx

        def compute_delu(i_begin, i_end, rhs, uf):
            for idx in range(i_begin, i_end):
                lti, lfi, lei = lt[idx], lf[idx], le[idx]

                for jdx in range(nvars):
                    ul = uf[lti][lfi, jdx, lei]
                    ur = rhs[jdx, idx]
                    du = ur - ul
                    uf[lti][lfi, jdx, lei] = du

        return self.be.make_loop(self.nfpts, compute_delu)

    def _make_pack(self):
        nvars = self.nvars
        lt, le, lf = self.rawlidx

        def pack(i_begin, i_end, lhs, uf):
            for idx in range(i_begin, i_end):
                lti, lfi, lei = lt[idx], lf[idx], le[idx]

                for jdx in range(nvars):
                    lhs[jdx, idx] = uf[lti][lfi, jdx, lei]

        return self.be.make_loop(self.nfpts, pack)

    def _sendrecv(self, mpifn, arr):
        # MPI Send or Receive init
        req = mpifn(arr, self._dest, self._tag)

        def start(q):
            # Function to save request in queue and start Send/Receive
            q.register(req)
            return req.Start()

        # Return Non-blocking send/recive and request (for finalise)
        return start, req

    def _make_send(self, arr):
        from mpi4py import MPI

        mpifn = MPI.COMM_WORLD.Send_init
        start, req = self._sendrecv(mpifn, arr)

        return start, req

    def _make_recv(self, arr):
        from mpi4py import MPI

        mpifn = MPI.COMM_WORLD.Recv_init
        start, req = self._sendrecv(mpifn, arr)

        return start, req


class BaseAdvecBCInters(BaseBCInters):
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

        # Get bc from `bcs.py` (in euler, navierstokes, rans...) and compile them
        self.bc = self._get_bc(self.be, bcf, bcc)

    def construct_kernels(self, elemap):
        self.construct_bc()

        # View of elemenet array
        self._fpts = fpts = tuple(cell.fpts for cell in elemap.values())

        if self.order > 1:
            # Kernel to compute differnce of solution at face
            self.compute_delu = Kernel(*self._make_delu(), fpts)
        else:
            self.compute_delu = NullKernel

    def _make_delu(self):
        nvars = self.nvars
        lidx = self.lidx
        nf = self.vec_snorm

        bc = self.bc
        array = self.be.local()

        def compute_delu(i_begin, i_end, lidx, nf, uf):
            for idx in range(i_begin, i_end):
                ur = array((nvars,), np.float64)
                nfi = nf[:, idx]

                lti, lei, lfi = lidx[:, idx]

                ul = uf[lti][lfi, :, lei]
                bc(ul, ur, nfi)

                for jdx in range(nvars):
                    du = ur[jdx] - ul[jdx]
                    uf[lti][lfi, jdx, lei] = du

        return self.be.make_loop(self.nfpts, compute_delu, lidx, nf)
