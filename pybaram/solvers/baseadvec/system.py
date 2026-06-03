# -*- coding: utf-8 -*-
from mpi4py import MPI
from pybaram.solvers.base.system import BaseSystem
from pybaram.solvers.baseadvec import BaseAdvecElements, BaseAdvecIntInters, BaseAdvecMPIInters, BaseAdvecBCInters, BaseAdvecVertex

import numpy as np


class BaseAdvecSystem(BaseSystem):
    name = 'baseadvec'
    _elements_cls = BaseAdvecElements
    _intinters_cls = BaseAdvecIntInters
    _bcinters_cls = BaseAdvecBCInters
    _mpiinters_cls = BaseAdvecMPIInters
    _vertex_cls = BaseAdvecVertex

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Get reciprocal of total volume for L2 residual normalization
        voli = sum(self.eles.tot_vol)
        self.rcp_vol = 1/self._comm.allreduce(voli, op=MPI.SUM)
        
    def rhside(self, idx_in=0, idx_out=1, t=0):
        # Adjust Banks
        self.eles.upts_in.idx = idx_in
        self.eles.upts_out.idx = idx_out

        # Queue for MPI
        q = self._queue

        # Compute solution at flux point (face center)
        self.eles.compute_fpts()

        if self.mpiint:
            # Start MPI communication for Inters
            self.mpiint.pack()
            self.mpiint.send(q)
            self.mpiint.recv(q)

        # Compute Difference of solution at Inters
        self.iint.compute_delu()
        self.bint.compute_delu()

        if self.mpiint:
            # Finalize MPI communication
            q.sync()

            # Compute Difference of solution at MPI Inters
            self.mpiint.compute_delu()

        # Compute extreme values at vertex
        self.vertex.compute_extv()

        if self.vertex.mpi:
            # Start MPI communication for Vertex
            self.vertex.pack()
            self.vertex.send(q)
            self.vertex.recv(q)

        # Compute gradient
        self.eles.compute_grad()

        if self.vertex.mpi:
            # Finalize MPI communication
            q.sync()

            # Unpack (Sort vetex extremes)
            self.vertex.unpack()

        # Compute slope limiter and reconstruction
        self.eles.compute_mlp_u()
        self.eles.compute_recon()

        if self._is_recon and self.mpiint:
            # Start MPI communication to exchange reconstructed values at face
            self.mpiint.pack()
            self.mpiint.send(q)
            self.mpiint.recv(q)

        # Compute flux
        self.iint.compute_flux()
        self.bint.compute_flux()

        if self.mpiint:
            # Finalize MPI communication
            q.sync()

            # Compute flux at MPI Inters
            self.mpiint.compute_flux()

        # Compute divergence
        self.eles.div_upts(t)

    def residual(self, idx_out=1, idx_res=0):
        # Adjust output bank
        self.eles.upts_res.idx = idx_out
        self.eles.resid_out.idx = idx_res

        # Compute residual from the current right hand side
        self.eles.compute_resid()
        
        # Collect L2-norm residual over all domains
        return self.reduce_residual()

    def reduce_residual(self):
        self.eles.reduce_resid()

        # Collect L2-norm residual over all domains
        self.be.wait()
        resid = self._comm.allreduce(sum(self.eles.h_resid), op=MPI.SUM)
        return np.sqrt(resid * self.rcp_vol)

    def spec_rad(self):
        # Compute solution at flux point (face center)
        self.eles.compute_fpts()

        # Compute spectral radius on faces
        self.iint.compute_spec_rad()
        self.bint.compute_spec_rad()

        if self.mpiint:
            self.mpiint.compute_spec_rad()

    def approx_jac(self):
        # Compute solution at flux point (face center)
        self.eles.compute_fpts()

        # Compute approximate Jacobian matrix on faces
        self.iint.compute_aprx_jac()
        self.bint.compute_aprx_jac()

        if self.mpiint:
            self.mpiint.compute_aprx_jac()

    def timestep(self, cfl, idx_in=0):
        # Compute time step with the given CFL number
        self.eles.upts_in.idx = idx_in
        self.eles.timestep(cfl)

    def post(self, idx_in=0):
        # Post-process
        self.eles.upts_in.idx = idx_in
        self.eles.post()

    def sum_reduce(a, b):
        return a + b
