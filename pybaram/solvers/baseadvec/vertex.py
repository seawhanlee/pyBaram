# -*- coding: utf-8 -*-
from pybaram.utils.misc import ProxyList
from pybaram.backends.types import (
    Kernel, NullKernel, MetaKernel, MPIPackKernel, MPIUnpackKernel,
    MPISendKernel
)
from pybaram.solvers.base import BaseVertex

import numpy as np


class BaseAdvecVertex(BaseVertex):
    _tag = 2314

    def make_array(self, limiter):
        if limiter == 'none':
            self.vpts = None
        else:
            if not hasattr(self, 'vpts'):
                # Allocate array to compute extremes at vertex
                self.vpts = self.be.alloc_array((2, self.nvars, self.nvtx))

        return self.vpts

    def construct_kernels(self, elemap):
        order = self.cfg.getint('solver', 'order', 1)
        limiter = self.cfg.get('solver', 'limiter', 'none')

        if order > 1 and limiter != 'none':
            if self._neivtx:
                # Construct kernels for MPI communication at vertex
                self.mpi = True
                self._construct_neighbors(self._neivtx)
            else:
                self.mpi = False

            # Kernel to compute exterems at vertex
            upts_in = tuple(ele.upts_in for ele in elemap.values())
            self.nele = len(upts_in)
            self.compute_extv = Kernel(*self._make_extv(), self.vpts, upts_in)
        else:
            self.compute_extv = NullKernel
            self.mpi = False

    def _make_extv(self):
        ivtx = self.be.convert_array(self._ivtx)
        # t, e, _ = self._idx
        t = self.be.convert_array(self._idx[0])
        e = self.be.convert_array(self._idx[1])
        nvars = self.nvars

        def cal_extv(i_begin, i_end, ivtx, t, e, vext, upts):
            for i in range(i_begin, i_end):
                for idx in range(ivtx[i], ivtx[i+1]):
                    ti, ei = t[idx], e[idx]
                    for jdx in range(nvars):
                        if idx == ivtx[i]:
                            vext[0, jdx, i] = upts[ti][jdx, ei]
                            vext[1, jdx, i] = upts[ti][jdx, ei]
                        else:
                            # Compute max / min solution at vertex
                            vext[0, jdx, i] = max(
                                vext[0, jdx, i], upts[ti][jdx, ei])
                            vext[1, jdx, i] = min(
                                vext[1, jdx, i], upts[ti][jdx, ei])

        return self.be.make_loop(self.nvtx, cal_extv, ivtx, t, e)

    def _construct_neighbors(self, neivtx):
        from mpi4py import MPI

        rawsbufs, rawrbufs = [], []
        sbufs, rbufs = [], []
        packs, unpacks = [], []
        sreqs, rreqs = [], []
        ivtxs = []

        nvars = self.nvars
        for p, v in neivtx.items():
            # Make buffer
            n = len(v)
            rawsbuf = self.be.alloc_array((2, nvars, n), pinned=True)
            rawrbuf = self.be.alloc_array((2, nvars, n), pinned=True)
            sbuf = self.be.convert_array(rawsbuf)
            rbuf = self.be.convert_array(rawrbuf)

            rawsbufs.append(rawsbuf)
            rawrbufs.append(rawrbuf)
            sbufs.append(sbuf)
            rbufs.append(rbuf)
            ivtxs.append(self.be.convert_array(v))

            packs.append(self._make_pack(n))
            unpacks.append(self._make_unpack(n))
            sreqs.append(self._make_send(rawsbuf, p))
            rreqs.append(self._make_recv(rawrbuf, p))

        def _communicate(reqs):
            def runall(q):
                # Start all MPI
                q.register(*reqs)
                MPI.Prequest.Startall(reqs)

            return runall

        # Start Sreqs (requsts for Send) and Rreqs (Request for Receive)
        self.send = MPISendKernel(self.be, _communicate(sreqs))
        self.recv = _communicate(rreqs)

        pack = MetaKernel([
            Kernel(pack[0], ivtx, self.vpts, buf)
            for pack, ivtx, buf in zip(packs, ivtxs, sbufs)
        ])
        unpack = MetaKernel([
            Kernel(unpack[0], ivtx, self.vpts, buf)
            for unpack, ivtx, buf in zip(unpacks, ivtxs, rbufs)
        ])

        self.rbufs = ProxyList(rbufs)

        # Sync Host <-> Device
        if self.be.name == 'cuda':
            dtoh = MetaKernel([
                Kernel(self.be.copy_array('d2h'), rawbuf, buf)
                for buf, rawbuf in zip(sbufs, rawsbufs)
            ])
            htod = MetaKernel([
                Kernel(self.be.copy_array('h2d'), buf, rawbuf)
                for rawbuf, buf in zip(rawrbufs, rbufs)
            ])
        else:
            dtoh = NullKernel()
            htod = NullKernel()

        self.pack = MPIPackKernel(self.be, pack, dtoh)
        self.unpack = MPIUnpackKernel(self.be, htod, unpack)
        self.pre_send = NullKernel()
        self.post_recv = NullKernel()

    def _make_pack(self, n):
        nvars = self.nvars

        def pack(i_begin, i_end, ivtx, vext, buf):
            for idx in range(i_begin, i_end):
                iv = ivtx[idx]
                for jdx in range(nvars):
                    # Save extremes to buffer
                    buf[0, jdx, idx] = vext[0, jdx, iv]
                    buf[1, jdx, idx] = vext[1, jdx, iv]

        return self.be.make_loop(n, pack)

    def _make_unpack(self, n):
        nvars = self.nvars

        def unpack(i_begin, i_end, ivtx, vext, buf):
            for idx in range(i_begin, i_end):
                iv = ivtx[idx]
                for jdx in range(nvars):
                    # Update extremes with exchanged values
                    vext[0, jdx, iv] = max(vext[0, jdx, iv], buf[0, jdx, idx])
                    vext[1, jdx, iv] = min(vext[1, jdx, iv], buf[1, jdx, idx])

        return self.be.make_loop(n, unpack)

    def _make_send(self, buf, dest):
        from mpi4py import MPI

        mpifn = MPI.COMM_WORLD.Send_init
        return mpifn(buf, dest, self._tag)

    def _make_recv(self, buf, dest):
        from mpi4py import MPI

        mpifn = MPI.COMM_WORLD.Recv_init
        return mpifn(buf, dest, self._tag)
