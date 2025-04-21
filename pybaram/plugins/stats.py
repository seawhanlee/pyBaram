# -*- coding: utf-8 -*-
from mpi4py import MPI
import time

from pybaram.plugins.base import BasePlugin, csv_write


class StatsPlugin(BasePlugin):
    name = 'stats'

    def __init__(self, intg, cfg, suffix):
        self.cfg = cfg

        sect = 'soln-plugin-{}'.format(self.name)
        self.flushsteps = cfg.getint(sect, 'flushsteps', 500)
        fname = cfg.get(sect, 'name', 'stats.csv')

        # Get rank
        self._rank = rank = MPI.COMM_WORLD.rank
        if rank == 0:
            # Out file name and header
            if not fname.endswith('.csv'):
                fname += '.csv'
            header = ['iter']

            # Make header
            if intg.mode == 'steady':
                ele = next(iter(intg.sys.eles))
                conservars = ele.conservars
                header += [*conservars, 'time']
                if intg.impl_op == 'approx-jacobian':
                    header += ['subiter', 'subres']
                self.t0 = time.time()
            else:
                header += ['t', 'dt']

            self.outf = csv_write(fname, header)

    def __call__(self, intg):
        if self._rank == 0:
            # Collect stats at this iteration
            stats = [intg.iter]

            if intg.mode == 'steady':
                # Compute time interval as millisecond unit
                interval = (time.time() - self.t0) * 1000.0
                resid = intg.resid / intg.resid0
                stats += [*resid.tolist(), interval]
                if intg.impl_op == 'approx-jacobian':
                    stats += [intg.subitnum, intg.subres]
                self.t0 = time.time()
            else:
                stats += [intg.tcurr, intg.dt]

            print(','.join(str(r) for r in stats), file=self.outf)

            # Check if stats are flushed or not or not at this iteration
            if intg.iter % self.flushsteps == 0:
                self.outf.flush()
