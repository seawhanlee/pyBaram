# -*- coding: utf-8 -*-
from mpi4py import MPI
from time import perf_counter

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
                header += [*conservars, 'cfl', 'time']
                if intg.impl_op == 'approx-jacobian':
                    header += ['subiter', 'subres']
            elif intg.mode == 'unsteady-dts':
                ele = next(iter(intg.sys.eles))
                conservars = ele.conservars
                header += ['t', *conservars, 'cfl', 'time']
            else:
                header += ['t', 'dt', 'time']

            self.t0 = perf_counter()
            self.outf = csv_write(fname, header)

    def _cfl(self, intg):
        if hasattr(intg, 'cfl'):
            return intg.cfl
        elif hasattr(intg, 'scfl'):
            return intg.scfl
        else:
            return ''

    def _interval(self):
        t1 = perf_counter()
        interval = (t1 - self.t0) * 1000.0
        self.t0 = t1

        return interval

    def __call__(self, intg):
        if self._rank == 0:
            # Collect stats at this iteration
            stats = [intg.iter]

            if intg.mode == 'steady':
                # Compute time interval as millisecond unit
                resid = intg.resid / intg.resid0
                stats += [*resid.tolist(), self._cfl(intg), self._interval()]
                if intg.impl_op == 'approx-jacobian':
                    stats += [intg.subitnum, intg.subres]
            elif intg.mode == 'unsteady-dts':
                interval = self._interval()
                last = len(intg.substats) - 1

                for i, substat in enumerate(intg.substats):
                    it, t, resid = substat
                    time = interval if i == last else ''
                    stats = [it, t, *resid.tolist(), self._cfl(intg), time]
                    print(','.join(str(r) for r in stats), file=self.outf)

                intg.substats.clear()
                if intg.iter % self.flushsteps == 0:
                    self.outf.flush()
                return
            else:
                stats += [intg.tcurr, intg.dt, self._interval()]

            print(','.join(str(r) for r in stats), file=self.outf)

            # Check if stats are flushed or not or not at this iteration
            if intg.iter % self.flushsteps == 0:
                self.outf.flush()
