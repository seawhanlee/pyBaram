# -*- coding: utf-8 -*-
from pybaram.solvers.baseadvec.system import BaseAdvecSystem
from pybaram.solvers.baseadvecdiff import BaseAdvecDiffElements, BaseAdvecDiffIntInters, BaseAdvecDiffMPIInters, BaseAdvecDiffBCInters


class BaseAdvecDiffSystem(BaseAdvecSystem):
    name = 'baseadvec'
    _elements_cls = BaseAdvecDiffElements
    _intinters_cls = BaseAdvecDiffIntInters
    _bcinters_cls = BaseAdvecDiffBCInters
    _mpiinters_cls = BaseAdvecDiffMPIInters

    def rhside(self, idx_in=0, idx_out=1, t=0):
        # Adjust Banks
        self.eles.upts_in.idx = idx_in
        self.eles.upts_out.idx = idx_out

        # Queue for MPI
        q = self._queue

        # Compute solution at flux point (face center)
        self.eles.compute_fpts()

        if self.mpiint:
            # Post receive early and prepare the send buffer.
            self.mpiint.recv(q)
            self.mpiint.pack()

        # Compute Difference of solution at Inters
        self.iint.compute_delu()
        self.bint.compute_delu()

        if self.mpiint:
            self.mpiint.send(q)

            # Finalize MPI communication
            q.sync()

            # Compute Difference of solution at MPI Inters
            self.mpiint.unpack()
            self.mpiint.compute_delu()

        # Compute extreme values at vertex
        self.vertex.compute_extv()

        if self.vertex.mpi:
            # Post receive early and prepare the send buffer.
            self.vertex.recv(q)
            self.vertex.pack()

        # Compute gradient
        self.eles.compute_grad()

        if self.vertex.mpi:
            self.vertex.send(q)

            # Finalize MPI communication
            q.sync()

            # Unpack (Sort vetex extremes)
            self.vertex.unpack()

        # Compute gradient at face
        self.iint.compute_grad_at()
        self.bint.compute_grad_at()

        if self.mpiint:
            # Post receive early and prepare the send buffer.
            self.mpiint.recv_grad(q)
            self.mpiint.pack_grad()

        # Compute slope limiter
        self.eles.compute_mlp_u()

        if self.mpiint:
            self.mpiint.send_grad(q)

            # Finalize MPI communication
            q.sync()

            # Compute gradient at MPI Inters
            self.mpiint.unpack_grad()
            self.mpiint.compute_grad_at()

        # Compute reconstruction
        self.eles.compute_recon()

        if self._is_recon and self.mpiint:
            # Post receive early and prepare the send buffer.
            self.mpiint.recv(q)
            self.mpiint.pack()

        # Compute flux
        self.iint.compute_flux()
        self.bint.compute_flux()

        if self.mpiint:
            if self._is_recon:
                self.mpiint.send(q)

            # Finalize MPI communication
            q.sync()

            # Compute flux at MPI Inters
            self.mpiint.unpack()
            self.mpiint.compute_flux()

        # Compute divergence 
        self.eles.div_upts(t)
