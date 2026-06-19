# -*- coding: utf-8 -*-
from pybaram.backends.types import Kernel, MetaKernel
from pybaram.inifile import INIFile
from pybaram.integrators.base import BaseIntegrator
from pybaram.integrators.relaxation import get_relaxation
from pybaram.utils.np import eps

import numpy as np
import re


class BaseSteadyIntegrator(BaseIntegrator):
    mode = 'steady'
    impl_op = 'none'

    def __init__(self, be, cfg, msh, soln, comm):
        # Get configurations for iterators
        self.itermax = cfg.getint('solver-time-integrator', 'max-iter')
        self.tol = cfg.getfloat('solver-time-integrator', 'tolerance')

        # Set current iteration
        if soln:
            # Get current iteration from previous result
            stats = INIFile()
            stats.fromstr(soln['stats'])
            self.iter = stats.getint('solver-time-integrator', 'iter', 0)
            if self.iter > 0:
                self.resid0 = np.array(stats.getlist(
                    'solver-time-integrator', 'resid0'))
        else:
            # Initialize iteration
            self.iter = 0

        # Indicator if solution is converted or not
        self.isconv = False

        super().__init__(be, cfg, msh, soln, comm)

        # Get CFL number
        self._cfl0 = cfg.getfloat('solver-time-integrator', 'cfl', 1.0)

        if cfg.has_section("solver-cfl-ramp"):
            sect = 'solver-cfl-ramp'

            # Get configuration of CFL linear ramp
            self._cfl_iter0 = cfg.getint(sect, 'iter0', self.itermax)
            self._cfl_itermax = cfg.getint(sect, 'max-iter', self.itermax)
            self._cflmax = cfg.getfloat(sect, 'max-cfl', self._cfl0)
        
            # Calculate the CFL increment for CFL ramp.
            if self._cfl_itermax > self._cfl_iter0:
                self._dcfl = (
                    (self._cflmax - self._cfl0)
                    / (self._cfl_itermax - self._cfl_iter0)
                )
                self._get_cfl = self._get_cfl_ramp
            else:
                self._get_cfl = self._get_cfl_const
        else:
            self._get_cfl = self._get_cfl_const

        # Adjust CFL for the RANS turbulent equations
        if self.sys.name.startswith('rans'):
            self._is_turb = True

            # Get turbulent CFL factor
            self._tcfl_fac = cfg.getfloat('solver-time-integrator', 'turb-cfl-factor', 1.0)
        else:
            self._is_turb = False

        # Specify residual variable for monitoring
        self.conservars = conservars = next(iter(self.sys.eles)).conservars
        rvar = cfg.get('solver-time-integrator', 'res-var', 'rho')
        self._res_idx = [i for i, e in enumerate(conservars) if e == rvar][0]

        # Construct kernels
        self.cfg = cfg
        self.construct_stages()

    @property
    def _cfl(self):
        return self._get_cfl()

    def _get_cfl_const(self):
        return self._cfl0

    def _get_cfl_ramp(self):
        # Return CFL considering CFL ramp
        if self.iter < self._cfl_iter0:
            return self._cfl0
        elif self.iter > self._cfl_itermax:
            return self._cflmax
        else:
            return self._cfl0 + self._dcfl*(self.iter - self._cfl_iter0)

    def run(self):
        # Run integrator until max iteration.
        if self.iter > self.itermax:
            raise ValueError(
                "Restart iteration {} exceeds max-iter {}".format(
                    self.iter, self.itermax
                )
            )

        if self.iter == self.itermax:
            self.isconv = False
            if self._show_final_status:
                print(
                    "Not converged : current iteration already equals "
                    "max-iter {}".format(
                        self.iter
                    )
                )
            return

        while self.iter < self.itermax:
            # Compute dt
            self._local_dt()

            # Compute one RK step
            self._curr_idx, resid = self.step()

            # Post actions after iteration
            self.complete_step(resid)

            # Check if tolerance is satisfied
            residual = (self.resid / self.resid0)
            if residual[self._res_idx] < self.tol:
                break

        # Fire off plugins
        self.isconv = True
        self.completed_handler(self)
        self.print_res(residual)

    def complete_step(self, resid):
        self.resid = resid

        # Check if reference residual (resid0) is existed or not
        if not hasattr(self, 'resid0'):
            # Avoid zero in resid0
            norm = self.cfg.get('solver-time-integrator', 'res-norm', 'True')
            if norm.lower() == 'no' or norm.lower() == 'false':
                self.resid0 = [1.0 for r in self.resid]
            else:
                self.resid0 = [r if r != 0 else eps for r in self.resid]

        self.iter += 1
        self.completed_handler(self)

    def _make_stages(self, out, *args):
        eq_str = self._make_stage_expr(args)

        if self._is_turb:
            # Substitute the 'dt' string with the dt array.
            eqf_str = re.sub('dt', 'dt[idx]', eq_str)
            eqt_str = re.sub('dt', '{}*dt[idx]'.format(self._tcfl_fac), eq_str)

            # Generate Python function for each RK stage
            f_txt =(
                f"def stage(i_begin, i_end, dt, upts):\n"
                f"    for idx in range(i_begin, i_end):\n"
                f"        for j in range(nfvars):\n"
                f"            upts[{out}][j, idx] = {eqf_str}\n"
                f"        for j in range(nfvars, nvars):\n"
                f"            upts[{out}][j, idx] = {eqt_str}"
            )
        else:
            # Substitute the 'dt' string with the dt array.
            eq_str = re.sub('dt', 'dt[idx]', eq_str)

            # Generate Python function for each RK stage
            f_txt =(
                f"def stage(i_begin, i_end, dt, upts):\n"
                f"  for idx in range(i_begin, i_end):\n"
                f"      for j in range(nvars):\n"
                f"          upts[{out}][j, idx] = {eq_str}"
            )

        kernels = []
        for ele in self.sys.eles:
            # Generate JIT kernel by looping RK stage function
            _stage = self._compile_stage(ele, f_txt)
            kernels.append(Kernel(*_stage, ele.dt, tuple(ele.upts)))
        
        # Collect RK stage kernels for elements
        return MetaKernel(kernels)

    def _local_dt(self):
        # Compute timestep of each cell using CFL
        self.cfl = self._cfl
        self.sys.timestep(self.cfl, self._curr_idx)

    def rhs(self, idx_in=0, idx_out=1):
        # Compute right hand side
        self.sys.rhside(idx_in, idx_out)

    def residual(self, idx_out=1):
        # Compute L2 norm residual
        return self.sys.residual(idx_out)

    def rhs_resid(self, idx_in=0, idx_out=1, **kwargs):
        self.sys.rhside(idx_in, idx_out)
        return self.sys.residual(idx_out)

    def print_res(self, residual):
        if not self._show_final_status:
            return

        # Print residual result
        idx = self._res_idx
        res = residual[idx]
        if res < self.tol:
            print("Converged : Residual of {} = {:05g} <= {:05g}".format(
                self.conservars[idx], res, self.tol))
        else:
            print("Not converged : Residual of {} = {:05g} > {:05g}".format(
                self.conservars[idx], res, self.tol))

    @property
    def _show_final_status(self):
        return not getattr(self, '_suppress_final_status', False)


class EulerExplicit(BaseSteadyIntegrator):
    name = 'eulerexplicit'
    nreg = 2

    def construct_stages(self):
        self._stages = [self._make_stages(0, 1, 0, 'dt', 1)]

    def step(self):
        stages = self._stages

        self.rhs(0, 1)
        resid = self.residual(1)
        stages[0]()

        self.sys.post(0)

        return 0, resid


class TVDRK3(BaseSteadyIntegrator):
    name = 'tvd-rk3'
    nreg = 3

    def construct_stages(self):
        self._stages = [
            self._make_stages(2, 1, 0, 'dt', 1),
            self._make_stages(2, 3/4, 0, 1/4, 2, 'dt/4', 1),
            self._make_stages(0, 1/3, 0, 2/3, 2, '2*dt/3', 1),
        ]

    def step(self):
        stages = self._stages

        self.rhs()
        stages[0]()
        self.sys.post(2)

        self.rhs(2, 1)
        stages[1]()
        self.sys.post(2)

        self.rhs(2, 1)
        resid = self.residual(1)
        stages[2]()
        self.sys.post(0)

        return 0, resid


class FiveStageRK(BaseSteadyIntegrator):
    """
    Jameson Multistage scheme
    ref : Blazek book 6.1.1 (Table 6.1)
    """
    name = 'rk5'
    nreg = 3

    def construct_stages(self):
        self._stages = [
            self._make_stages(2, 1, 0, '0.0533*dt', 1),
            self._make_stages(2, 1, 0, '0.1263*dt', 1),
            self._make_stages(2, 1, 0, '0.2375*dt', 1),
            self._make_stages(2, 1, 0, '0.4414*dt', 1),
            self._make_stages(0, 1, 0, 'dt', 1)
        ]

    def step(self):
        stages = self._stages

        self.rhs()
        stages[0]()
        self.sys.post(2)
        
        self.rhs(2, 1)
        stages[1]()
        self.sys.post(2)

        self.rhs(2, 1)
        stages[2]()
        self.sys.post(2)

        self.rhs(2, 1)
        stages[3]()
        self.sys.post(2)

        self.rhs(2, 1)
        resid = self.residual(1)
        stages[4]()
        self.sys.post(0)

        return 0, resid
      

class SteadyRelaxationIntegrator(BaseSteadyIntegrator):
    name = None
    nreg = 2

    def construct_stages(self):
        self._rhs_idx = 1
        self.relaxation = get_relaxation(
            self.cfg, self, 'solver-time-integrator', name=self.name
        )
        self.relaxation.build(0.0)

    def step(self):
        return self.relaxation.step()


class LUSGS(SteadyRelaxationIntegrator):
    name = 'lu-sgs'
    impl_op = 'spectral-radius'


class ColoredLUSGS(SteadyRelaxationIntegrator):
    name = 'colored-lu-sgs'
    impl_op = 'spectral-radius'


class BlockLUSGS(SteadyRelaxationIntegrator):
    name = 'blu-sgs'
    impl_op = 'approx-jacobian'


class ColoredBlockLUSGS(SteadyRelaxationIntegrator):
    name = 'colored-blu-sgs'
    impl_op = 'approx-jacobian'


class PETSc(SteadyRelaxationIntegrator):
    name = 'petsc'
    impl_op = 'approx-jacobian'
