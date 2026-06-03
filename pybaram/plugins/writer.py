# -*- coding: utf-8 -*-
from mpi4py import MPI

import h5py
import numpy as np

from pybaram.inifile import INIFile
from pybaram.plugins.base import BasePlugin


class WriterPlugin(BasePlugin):
    # Plugin to write output file 
    name = 'writer'

    def __init__(self, intg, cfg, suffix):
        super().__init__(intg, cfg, suffix)

        # Parse uuid
        uuid = intg.mesh['mesh_uuid']

        # Parse conservative variable name
        self._prefix = prefix = 'soln'
        vars = ','.join(next(iter(intg.sys.eles)).conservars)

        # Assign prefix and field variable name
        self._stat = stat = INIFile()
        stat.set('data', 'prefix', prefix)
        stat.set('data', 'fields', vars)

        # Extract configuration and save
        self._fname = cfg.get('soln-plugin-writer', 'name')
        self.out = {'config': cfg.tostr(), 'mesh_uuid': uuid}

        # Check integrator mode (steady | unsteady) and frequency to compute the plugin
        self.mode = mode = intg.mode
        if mode in ('unsteady', 'unsteady-dts'):
            self.dtout = cfg.getfloat('soln-plugin-writer', 'dt-out')
            self.tout_next = intg.tcurr + self.dtout

            intg.add_tlist(self.dtout)
        else:
            self.itout = cfg.getint('soln-plugin-writer', 'iter-out', 100)

        # get MPI_COMM_WORLD and rank
        self._comm = comm = MPI.COMM_WORLD
        self._rank = rank = comm.rank

        # Collect solution info
        etypes = [ele.name for ele in intg.sys.eles]
        shapes = [sol.shape for sol in intg.curr_soln]
        self._soln_fields = [self._prefix] + [
            '{}_{}'.format(self._prefix, name)
            for name, idx in intg.restart_soln_idxs()
        ]
        eleinfo = comm.gather(
            tuple((e, s) for e, s in zip(etypes, shapes)), root=0
            )

        if intg.is_aux:
            # Collect aux variable infos
            ashapes = [aux.shape for aux in intg.curr_aux]
            auxinfo = comm.gather(
                tuple((e, s) for e, s in zip(etypes, ashapes)), root=0
                )

        if rank == 0:
            # Buffers and request for serial write
            self._mpi_bufs = mpi_bufs = []
            self._mpi_reqs = mpi_reqs = []
            self._mpi_keys = mpi_keys = []
            self._loc_keys = loc_keys = []

            # Predefine receive requests
            for p, eleinfo_p in enumerate(eleinfo):
                tag = 0
                for field in self._soln_fields:
                    for etype, shape in eleinfo_p:
                        key = '{}_{}_p{}'.format(field, etype, p)

                        if p == 0:
                            loc_keys.append(key)
                        else:
                            buf = np.empty(shape, dtype=np.float64)
                            req = comm.Recv_init(buf, p, tag)

                            mpi_bufs.append(buf)
                            mpi_reqs.append(req)
                            mpi_keys.append(key)

                        tag += 1

            if intg.is_aux:
                for p, auxinfo_p in enumerate(auxinfo):
                    tag_offset = len(self._soln_fields)*len(eleinfo[p])
                    for tag, (etype, shape) in enumerate(auxinfo_p, tag_offset):

                        key = 'aux_{}_p{}'.format(etype, p)

                        if p == 0:
                            loc_keys.append(key)
                        else:
                            buf = np.empty(shape, dtype=np.float64)
                            req = comm.Recv_init(buf, p, tag)

                            mpi_bufs.append(buf)
                            mpi_reqs.append(req)
                            mpi_keys.append(key)

        if intg.iter == 0:
            # Save initial field
            self(intg)           

    def __call__(self, intg):
        if self.mode in ('unsteady', 'unsteady-dts'):
            if abs(intg.tcurr - self.tout_next) > 1e-6:
                return

            # Save stats of unsteady simulation
            t = intg.tcurr
            self._stat.set('solver-time-integrator', 'tcurr', str(t))
            self._stat.set('solver-time-integrator', 'iter', str(intg.iter))
            if self.mode == 'unsteady-dts':
                self._stat.set('solver-time-integrator', 'piter', str(intg.piter))

            self.tout_next += self.dtout

            fname = self._fname.format(n=0, t=t) + '.pbrs'
        else:
            if not intg.isconv and intg.iter % self.itout:
                return

            # Save stats of steady simulation (iteration, resid0)
            self._stat.set('solver-time-integrator', 'iter', str(intg.iter))
            if hasattr(intg, 'resid0') and not hasattr(self, '_resid0'):
                self._resid0 = resid0 = intg.resid0
                resid0_txt = ','.join(str(v) for v in resid0)
                self._stat.set('solver-time-integrator', 'resid0', resid0_txt)

            fname = self._fname.format(n=intg.iter, t=0) + '.pbrs'

        # Save stats
        self.out['stats'] = self._stat.tostr()

        if self._rank != 0:
            # Send data to rank=0
            tag = 0
            for solns in self._soln_arrays(intg):
                for soln in solns:
                    self._comm.Send(soln.copy(), 0, tag)
                    tag += 1

            if intg.is_aux:
                for aux in intg.curr_aux:
                    self._comm.Send(aux.copy(), 0, tag)
                    tag += 1
        else:
            # Local data
            curr = []
            for solns in self._soln_arrays(intg):
                curr += solns

            if intg.is_aux:
                curr += intg.curr_aux

            # Save loacl data
            for key, data in zip(self._loc_keys, curr):
                self.out[key] = data

            # Communicate from other processors
            MPI.Prequest.Startall(self._mpi_reqs)
            MPI.Prequest.Waitall(self._mpi_reqs)

            # Save data from other ranks
            for key, buf in zip(self._mpi_keys, self._mpi_bufs):
                self.out[key] = buf

            # Write data
            with h5py.File(fname, 'w') as f:
                for k, v in self.out.items():
                    f[k] = v

    def _soln_arrays(self, intg):
        solns = [intg.curr_soln]
        solns += [
            intg.soln_at(idx) for name, idx in intg.restart_soln_idxs()
        ]

        return solns
