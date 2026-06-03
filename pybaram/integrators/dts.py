# -*- coding: utf-8 -*-
from pybaram.backends.types import Kernel
from pybaram.inifile import INIFile
from pybaram.integrators.base import BaseIntegrator
from pybaram.integrators.relaxation import get_relaxation
from pybaram.utils.misc import ProxyList
from pybaram.utils.np import eps

import numpy as np


class BaseDTSIntegrator(BaseIntegrator):
    """
    Dual-time stepping integrator with fixed physical time step.
    """
    name = None
    mode = 'unsteady-dts'

    def __init__(self, be, cfg, msh, soln, comm):
        self._init_banks()

        names = ','.join(name for name, idx in self.restart_soln_idxs())
        idxs = ','.join(str(idx) for name, idx in self.restart_soln_idxs())
        cfg.set('solver-time-integrator-dts', 'restart-soln-names', names)
        cfg.set('solver-time-integrator-dts', 'restart-soln-idxs', idxs)

        # Physical-time range is given as (start, end).
        tcurr, tend_cfg = eval(cfg.get('solver-time-integrator', 'time'))
        self.tstart = tcurr
        self.dt = dt = cfg.getfloat('solver-time-integrator', 'dt')

        if dt <= 0:
            raise ValueError("solver-time-integrator.dt must be positive")

        if tend_cfg < tcurr:
            raise ValueError("solver-time-integrator.time end must be >= start")

        nsteps = int(np.ceil((tend_cfg - tcurr)/dt - eps))
        self.tend = tend = tcurr + nsteps*dt
        self.tlist = tcurr + dt*np.arange(1, nsteps + 1)

        # Number of completed physical time steps.  This is separate from
        # `iter`, which counts pseudo-time sub-iterations.
        piter = None

        # Set current iteration
        if soln:
            stats = INIFile()
            stats.fromstr(soln['stats'])

            if stats.has_option('solver-time-integrator', 'tcurr'):
                self.tcurr = stats.getfloat('solver-time-integrator', 'tcurr')
                self.iter = stats.getint('solver-time-integrator', 'iter', 0)
                # Preserve the BDF startup state across restart.  Without
                # this, a restarted BDF2/BDF3 run would fall back to BDF1.
                if stats.has_option('solver-time-integrator', 'piter'):
                    piter = stats.getint('solver-time-integrator', 'piter')
                self.tlist = self.tlist[self.tlist > self.tcurr + eps]
            else:
                self.tcurr = tcurr
                self.iter = 0
        else:
            self.tcurr = tcurr
            self.iter = 0

        if piter is None:
            # Older restart files do not have `piter`; infer it from time.
            piter = int(round((self.tcurr - self.tstart)/dt))

        self.piter = max(0, piter)

        # Get Relaxation
        self.relaxation = get_relaxation(cfg, self, 'solver-time-relaxation')

        super().__init__(be, cfg, msh, soln, comm)

        # Controls for sub-iterations.
        self.scfl = cfg.getfloat('solver-time-integrator', 'sub-cfl', 1.0)
        self.sitermax = cfg.getint('solver-time-integrator', 'sub-iter', 50)
        self.subtol = cfg.getfloat('solver-time-integrator', 'sub-tol', 0.001)

        if self.sys.name.startswith('rans'):
            self._is_turb = True
            self._tcfl_fac = cfg.getfloat(
                'solver-time-integrator', 'turb-cfl-factor', 1.0
            )
        else:
            self._is_turb = False

        self.conservars = conservars = next(iter(self.sys.eles)).conservars
        rvar = cfg.get('solver-time-integrator', 'res-var', 'rho')
        self._res_idx = [i for i, e in enumerate(conservars) if e == rvar][0]

        self.construct_stages()

    def _init_banks(self):
        pass

    def restart_soln_idxs(self):
        return [
            ('prev{}'.format(i + 1), idx)
            for i, idx in enumerate(self._prev_idxs)
        ]

    def add_tlist(self, dt):
        # DTS advances on a fixed physical timestep.  Plugin output intervals
        # therefore need to line up with completed physical steps.
        ratio = dt / self.dt
        multiple = round(ratio)

        if multiple < 1 or not np.isclose(ratio, multiple, rtol=1e-12, atol=1e-12):
            raise ValueError(
                "dt-out for dual-time stepping must be a positive integer "
                "multiple of solver-time-integrator.dt"
            )

    def run(self):
        for t in self.tlist:
            self.advance_to(t)

    def _local_dtau(self):
        # Compute pseudo time step for each element
        self.sys.timestep(self.scfl, self._curr_idx)

    def _make_bdf_source(self, out, *args):
        kernels = []
        
        eq_str = self._make_stage_expr(args)
        f_txt = (
            f"def stage(i_begin, i_end, upts):\n"
            f"    for idx in range(i_begin, i_end):\n"
            f"        for j in range(nvars):\n"
            f"            upts[{out}][j, idx] = {eq_str}\n"
        )
        sidxs = args[1::2][::-1]
        for s1, s2 in zip(sidxs[:-1], sidxs[1:]):
            f_txt += f"            upts[{s1}][j, idx] = upts[{s2}][j, idx]\n"

        for ele in self.sys.eles:
            # Generate JIT kernel by looping source function
            _kern = self._compile_stage(ele, f_txt)
            kernels.append(Kernel(*_kern, tuple(ele.upts)))

        return ProxyList(kernels)

    def _make_add_source(self, out, *args):
        kernels = []

        eq_str = self._make_stage_expr(args)
        f_txt = (
            f"def stage(i_begin, i_end, upts):\n"
            f"    for idx in range(i_begin, i_end):\n"
            f"        for j in range(nvars):\n"
            f"            upts[{out}][j, idx] -= {eq_str}\n"
        )

        for ele in self.sys.eles:
            # Generate JIT kernel by looping source function
            _kern = self._compile_stage(ele, f_txt)
            kernels.append(Kernel(*_kern, tuple(ele.upts)))
        return ProxyList(kernels)

    def rhs_resid(self, idx_in=None, idx_out=None, t=0):
        self.sys.rhside(idx_in, idx_out, t=t)
        self._add_source()

        return self.sys.residual(idx_out)

    def advance_to(self, ttag):
        # Use lower-order BDF while the required solution history is not yet
        # available: BDF2 starts as BDF1, and BDF3 starts as BDF1 then BDF2.
        self._set_active_order(min(self.piter + 1, self._target_order))

        # Calculate source term
        self._source()
        self.substats = []

        # Sub-iterations
        for it in range(self.sitermax):
            # Compute dtau
            self._local_dtau()

            # Compute one relaxation
            self._curr_idx, resid = self.step(ttag)

            # Check if converged or not
            residi = resid[self._res_idx]
            if it == 0:
                resid0 = residi if residi != 0 else eps
                subresid = 1.0
            else:
                subresid = residi/resid0

            self.subitnum = it + 1
            self.subres = subresid
            self.iter += 1
            self.substats.append((self.iter, ttag, resid.copy()))

            if subresid < self.subtol:
                break

        # Post
        self.tcurr = ttag
        # Completed one physical step.  Pseudo-time iterations are tracked by
        # `iter` inside the sub-iteration loop.
        self.piter += 1
        self.completed_handler(self)


class BDFDTSIntegrator(BaseDTSIntegrator):
    name = None
    _coeffs = None
    _coeffs_by_order = {
        1: (1.0, -1.0),
        2: (1.5, -2.0, 0.5),
        3: (11/6, -3.0, 1.5, -1/3)
    }

    def _init_banks(self):
        self._target_order = order = len(self._coeffs) - 1

        self._curr_idx = 0
        self._rhs_idx = 1
        self._prev_idxs = list(range(2, order + 1))
        self._src_idx = order + 1

    def construct_stages(self):
        dt = self.dt

        # Pre-build kernels for each startup order.  The active set is swapped
        # as physical-time history becomes available.
        self._stages = {}
        self._active_order = None

        for order in range(1, self._target_order + 1):
            _c = self._coeffs_by_order[order]

            source_args = []
            for c, idx in zip(_c[1:], [self._curr_idx, *self._prev_idxs]):
                source_args.extend([c/dt, idx])

            self._stages[order] = (
                self._make_bdf_source(self._src_idx, *source_args),
                self._make_add_source(
                    self._rhs_idx, _c[0]/dt, self._curr_idx, 1.0, self._src_idx
                ),
                _c[0]/dt
            )

        self._set_active_order(min(self.piter + 1, self._target_order))

    def _set_active_order(self, order):
        if order == self._active_order:
            return

        # The BDF diagonal coefficient changes with order, so relaxation
        # kernels must be rebuilt when the active BDF order changes.
        self._source, self._add_source, a0 = self._stages[order]
        self.relaxation.build(a0)
        self._active_order = order

    def step(self, t):
        return self.relaxation.step(t=t)


class BDF1DTSIntegrator(BDFDTSIntegrator):
    name = 'bdf1'
    nreg = 3
    _coeffs = (1.0, -1.0)


class BDF2DTSIntegrator(BDFDTSIntegrator):
    name = 'bdf2'
    nreg = 4
    _coeffs = (1.5, -2, 0.5)


class BDF3DTSIntegrator(BDFDTSIntegrator):
    name = 'bdf3'
    nreg = 5
    _coeffs = (11/6, -3.0, 1.5, -1/3)
