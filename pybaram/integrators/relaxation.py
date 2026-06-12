# -*- coding: utf-8 -*-
from pybaram.backends.types import ArrayBank, Kernel, MetaKernel
from pybaram.utils.misc import subclass_by_name
from pybaram.utils.np import eps

import numpy as np


def get_relaxation(cfg, intg, sect, name=None, default='lu-sgs'):
    if name is None:
        if cfg.has_section(sect):
            name = cfg.get(sect, 'method', default)
        else:
            name = cfg.get('solver-time-integrator', 'relaxation', default)

    return subclass_by_name(BaseRelaxation, name)(intg, cfg, sect)


class BaseRelaxation:
    name = None
    impl_op = None

    def __init__(self, intg, cfg, sect):
        intg.impl_op = self.impl_op
        self._intg = intg
        self._cfg = cfg

        # Configuration section for options owned by the relaxation solver.
        self._sect = sect


class BaseLUSGSRelaxation(BaseRelaxation):
    impl_op = 'spectral-radius'

    def step(self, **kwargs):
        intg = self._intg

        resid = intg.rhs_resid(0, 1, **kwargs)

        intg.sys.eles.lusgs()
        intg.sys.eles.update()
        intg.sys.post(0)

        return 0, resid


class LUSGSRelaxation(BaseLUSGSRelaxation):
    name = 'lu-sgs'

    def build(self, a0, kappa=1.01):
        from pybaram.integrators.lusgs import (
            make_lusgs_common, make_lusgs_update, make_serial_lusgs
        )

        intg = self._intg
        be = intg.be
        idx_u = intg._curr_idx
        idx_du = intg._rhs_idx

        for ele in intg.sys.eles:
            diag = ele.fpts[1, 0]
            fnorm_vol, vec_fnorm = ele.fnorm_vol, ele.vec_fnorm
            nei_ele = ele.nei_ele

            def make_kernels(nv, flux, lambdaf, factor=1.0):
                pre_lusgs = make_lusgs_common(
                    ele, a0=a0, factor=factor, kappa=kappa
                )
                lsweep, usweep = make_serial_lusgs(be, ele, nv, flux)

                return (
                    Kernel(
                        *be.make_loop(ele.neles, pre_lusgs, fnorm_vol),
                        ele.dt, diag, lambdaf
                    ),
                    Kernel(
                        *be.make_loop(ele.neles, lsweep, fnorm_vol, vec_fnorm, nei_ele),
                        ele.upts[idx_u], ele.upts[idx_du], diag, ele.dsrc, lambdaf
                    ),
                    Kernel(
                        *be.make_loop(ele.neles, usweep, fnorm_vol, vec_fnorm, nei_ele),
                        ele.upts[idx_u], ele.upts[idx_du], diag, ele.dsrc, lambdaf
                    )
                )

            pre_lusgs, lsweep, usweep = make_kernels(
                (0, ele.nfvars), ele.flux_container(), ele.fspr
            )
            kernels = [pre_lusgs, lsweep, usweep]

            if intg._is_turb:
                pre_tlusgs, tlsweep, tusweep = make_kernels(
                    (ele.nfvars, ele.nvars),
                    ele.tflux_container(),
                    ele.tfspr,
                    factor=intg._tcfl_fac
                )
                kernels += [pre_tlusgs, tlsweep, tusweep]

            ele.lusgs = MetaKernel(kernels)

            update = make_lusgs_update(ele)
            ele.update = Kernel(
                *be.make_loop(ele.neles, update),
                ele.upts[idx_u], ele.upts[idx_du]
            )


class ColoredLUSGSRelaxation(BaseLUSGSRelaxation):
    name = 'colored-lu-sgs'

    def build(self, a0, kappa=1.01):
        from pybaram.integrators.lusgs import (
            make_colored_lusgs, make_lusgs_common, make_lusgs_update
        )

        intg = self._intg
        be = intg.be
        idx_u = intg._curr_idx
        idx_du = intg._rhs_idx

        for ele in intg.sys.eles:
            ncolor, _icolor, _lev_color = ele.coloring()
            icolor = be.convert_array(_icolor)
            lev_color = be.convert_array(_lev_color)

            fnorm_vol = be.convert_array(ele.fnorm_vol)
            vec_fnorm = be.convert_array(ele.vec_fnorm)
            nei_ele = be.convert_array(ele.nei_ele)
            diag = ele.fpts[1, 0]

            def make_kernels(nv, flux, lambdaf, factor=1.0):
                pre_lusgs = make_lusgs_common(
                    ele, a0=a0, factor=factor, kappa=kappa
                )
                lsweep, usweep = make_colored_lusgs(be, ele, nv, flux)

                pre_lusgs = Kernel(
                    *be.make_loop(ele.neles, pre_lusgs, fnorm_vol),
                    ele.dt, diag, lambdaf
                )

                lsweeps = [
                    Kernel(
                        *be.make_loop(
                            ne, lsweep, fnorm_vol, vec_fnorm, nei_ele,
                            icolor, lev_color, n0=n0
                        ),
                        ele.upts[idx_u], ele.upts[idx_du], diag, ele.dsrc,
                        lambdaf
                    )
                    for n0, ne in zip(ncolor[:-1], ncolor[1:])
                ]

                usweeps = [
                    Kernel(
                        *be.make_loop(
                            ne, usweep, fnorm_vol, vec_fnorm, nei_ele,
                            icolor, lev_color, n0=n0
                        ),
                        ele.upts[idx_u], ele.upts[idx_du], diag, ele.dsrc,
                        lambdaf
                    )
                    for n0, ne in zip(ncolor[::-1][1:], ncolor[::-1][:-1])
                ]

                return pre_lusgs, lsweeps, usweeps

            pre_lusgs, lsweeps, usweeps = make_kernels(
                (0, ele.nfvars), ele.flux_container(), ele.fspr
            )
            kernels = [pre_lusgs, *lsweeps, *usweeps]

            if intg._is_turb:
                pre_tlusgs, tlsweeps, tusweeps = make_kernels(
                    (ele.nfvars, ele.nvars),
                    ele.tflux_container(),
                    ele.tfspr,
                    factor=intg._tcfl_fac
                )
                kernels += [pre_tlusgs, *tlsweeps, *tusweeps]

            ele.lusgs = MetaKernel(kernels)

            update = make_lusgs_update(ele)
            ele.update = Kernel(
                *be.make_loop(ele.neles, update),
                ele.upts[idx_u], ele.upts[idx_du]
            )


class BaseBlockLUSGSRelaxation(BaseRelaxation):
    impl_op = 'approx-jacobian'

    def _init_subiteration_controls(self):
        # Block LU-SGS subiteration controls.
        self.subiter = self._cfg.getint(self._sect, 'sub-iter', 10)
        self.subrtol = self._cfg.getfloat(self._sect, 'sub-rtol', 0.1)
        self.subatol = self._cfg.getfloat(self._sect, 'sub-atol', 0.0)

    def _make_update_kernels(self, ele, make_blusgs_update, make_sub_residual):
        intg = self._intg
        be = intg.be
        idx_u = intg._curr_idx

        update = make_blusgs_update(ele)
        ele.update = Kernel(
            *be.make_loop(ele.neles, update), ele.upts[idx_u], ele.du
        )

        subres = make_sub_residual(ele)
        ele.subresid = Kernel(
            *be.make_loop(ele.neles, subres), ele.vol, ele.du, ele.dup,
            ele.resid_out
        )

    def step(self, **kwargs):
        intg = self._intg

        resid = intg.rhs_resid(0, 1, **kwargs)

        # Reset correction histories.
        intg.sys.eles.du.set(0)
        intg.sys.eles.dup.set(0)

        # Compute diagonal matrix
        intg.sys.eles.pre_blusgs()
        subresid = 1.0

        # Block LU-SGS subiterations.
        for it in range(self.subiter):
            # Block LU-SGS sweep
            intg.sys.eles.blusgs_sweep()

            # Compute sub-residual from all elements
            intg.sys.eles.subresid()
            drho = intg.sys.reduce_residual()[intg._res_idx]

            # Check sub-convergence
            if drho <= self.subatol:
                subresid = 0.0
                break
            elif it == 0:
                drho1 = drho if drho != 0 else eps
            else:
                subresid = drho/drho1
                if subresid < self.subrtol:
                    break

        intg.sys.eles.update()
        intg.sys.post(0)
        intg.subitnum = it + 1
        intg.subres = subresid

        return 0, resid


class BlockLUSGSRelaxation(BaseBlockLUSGSRelaxation):
    name = 'blu-sgs'

    def build(self, a0):
        from pybaram.integrators.blusgs import (
            make_blusgs_update, make_pre_blusgs, make_serial_blusgs,
            make_sub_residual, make_tpre_blusgs
        )

        intg = self._intg
        be = intg.be

        self._init_subiteration_controls()
        idx_u = intg._curr_idx
        idx_rhs = intg._rhs_idx

        for ele in intg.sys.eles:
            fnorm_vol = be.convert_array(ele.fnorm_vol)
            nei_ele = be.convert_array(ele.nei_ele)
            diag = be.alloc_array((ele.nfvars, ele.nfvars, ele.neles))

            ele.du = ArrayBank(ele.fpts, 1)
            ele.dup = ArrayBank(ele.fpts, 2)

            nv = (0, ele.nfvars)
            pre_blusgs = make_pre_blusgs(be, ele, nv, a0=a0)
            lower, upper = make_serial_blusgs(be, ele, nv)

            pre_blusgs = Kernel(
                *be.make_loop(ele.neles, pre_blusgs, fnorm_vol),
                ele.dt, diag, ele.jmat
            )
            lsweep = Kernel(
                *be.make_loop(ele.neles, lower, fnorm_vol, nei_ele),
                ele.upts[idx_rhs], ele.du, diag, ele.jmat
            )
            usweep = Kernel(
                *be.make_loop(ele.neles, upper, fnorm_vol, nei_ele),
                ele.upts[idx_rhs], ele.du, diag, ele.jmat
            )

            pre_kernels = [pre_blusgs]
            sweep_kernels = [lsweep, usweep]

            if intg._is_turb:
                tdiag = be.alloc_array((ele.nturbvars, ele.nturbvars, ele.neles))
                tnv = (ele.nfvars, ele.nvars)

                srcjacobian = ele.make_source_jacobian()
                pre_tblusgs = make_tpre_blusgs(
                    be, ele, tnv, srcjacobian, intg._tcfl_fac, a0=a0
                )
                pre_tblusgs = Kernel(
                    *be.make_loop(ele.neles, pre_tblusgs, fnorm_vol),
                    ele.upts[idx_u], ele.dt, tdiag, ele.tjmat, ele.dsrc
                )

                tlower, tupper = make_serial_blusgs(be, ele, tnv)
                tlsweep = Kernel(
                    *be.make_loop(ele.neles, tlower, fnorm_vol, nei_ele),
                    ele.upts[idx_rhs], ele.du, tdiag, ele.tjmat
                )
                tusweep = Kernel(
                    *be.make_loop(ele.neles, tupper, fnorm_vol, nei_ele),
                    ele.upts[idx_rhs], ele.du, tdiag, ele.tjmat
                )

                pre_kernels += [pre_tblusgs]
                sweep_kernels += [tlsweep, tusweep]

            ele.pre_blusgs = MetaKernel(pre_kernels)
            ele.blusgs_sweep = MetaKernel(sweep_kernels)

            self._make_update_kernels(
                ele, make_blusgs_update, make_sub_residual
            )


class ColoredBlockLUSGSRelaxation(BaseBlockLUSGSRelaxation):
    name = 'colored-blu-sgs'

    def build(self, a0):
        from pybaram.integrators.blusgs import (
            make_blusgs_update, make_colored_blusgs, make_pre_blusgs,
            make_sub_residual, make_tpre_blusgs
        )

        intg = self._intg
        be = intg.be

        self._init_subiteration_controls()
        idx_u = intg._curr_idx
        idx_rhs = intg._rhs_idx

        for ele in intg.sys.eles:
            ncolor, _icolor, _lev_color = ele.coloring()
            icolor = be.convert_array(_icolor)
            lev_color = be.convert_array(_lev_color)

            fnorm_vol = be.convert_array(ele.fnorm_vol)
            nei_ele = be.convert_array(ele.nei_ele)
            diag = be.alloc_array((ele.nfvars, ele.nfvars, ele.neles))

            ele.du = ArrayBank(ele.fpts, 1)
            ele.dup = ArrayBank(ele.fpts, 2)

            nv = (0, ele.nfvars)
            pre_blusgs = make_pre_blusgs(be, ele, nv, a0=a0)
            sweep = make_colored_blusgs(be, ele, nv)

            pre_blusgs = Kernel(
                *be.make_loop(ele.neles, pre_blusgs, fnorm_vol),
                ele.dt, diag, ele.jmat
            )
            lsweeps = [
                Kernel(
                    *be.make_loop(ne, sweep, fnorm_vol, nei_ele,
                                  icolor, lev_color, n0=n0),
                    ele.upts[idx_rhs], ele.du, diag, ele.jmat
                )
                for n0, ne in zip(ncolor[:-1], ncolor[1:])
            ]
            usweeps = [
                Kernel(
                    *be.make_loop(ne, sweep, fnorm_vol, nei_ele,
                                  icolor, lev_color, n0=n0),
                    ele.upts[idx_rhs], ele.du, diag, ele.jmat
                )
                for n0, ne in zip(ncolor[::-1][1:], ncolor[::-1][:-1])
            ]

            pre_kernels = [pre_blusgs]
            sweep_kernels = [*lsweeps, *usweeps]

            if intg._is_turb:
                ele.tdiag = be.alloc_array(
                    (ele.nturbvars, ele.nturbvars, ele.neles)
                )
                tnv = (ele.nfvars, ele.nvars)

                srcjacobian = ele.make_source_jacobian()
                pre_tblusgs = make_tpre_blusgs(
                    be, ele, tnv, srcjacobian, intg._tcfl_fac, a0=a0
                )
                pre_tblusgs = Kernel(
                    *be.make_loop(ele.neles, pre_tblusgs, fnorm_vol),
                    ele.upts[idx_u], ele.dt, ele.tdiag, ele.tjmat, ele.dsrc
                )

                tsweep = make_colored_blusgs(be, ele, tnv)
                tlsweeps = [
                    Kernel(
                        *be.make_loop(ne, tsweep, fnorm_vol, nei_ele,
                                      icolor, lev_color, n0=n0),
                        ele.upts[idx_rhs], ele.du, ele.tdiag, ele.tjmat
                    )
                    for n0, ne in zip(ncolor[:-1], ncolor[1:])
                ]
                tusweeps = [
                    Kernel(
                        *be.make_loop(ne, tsweep, fnorm_vol, nei_ele,
                                      icolor, lev_color, n0=n0),
                        ele.upts[idx_rhs], ele.du, ele.tdiag, ele.tjmat
                    )
                    for n0, ne in zip(ncolor[::-1][1:], ncolor[::-1][:-1])
                ]

                pre_kernels += [pre_tblusgs]
                sweep_kernels += [*tlsweeps, *tusweeps]

            ele.pre_blusgs = MetaKernel(pre_kernels)
            ele.blusgs_sweep = MetaKernel(sweep_kernels)

            self._make_update_kernels(
                ele, make_blusgs_update, make_sub_residual
            )


class PETScRelaxation(BaseRelaxation):
    """Per-element PETSc KSP relaxation.

    Each element group builds a partition-local BSR system from the existing
    face Jacobian storage, solves it with PETSc KSP, and scatters the solution
    back to the SOA correction buffer ele.du.
    """

    name = 'petsc'
    impl_op = 'approx-jacobian'

    def build(self, a0):
        from pybaram.integrators.blusgs import make_blusgs_update
        from pybaram.integrators.petsc import (
            PETScElementSolve, make_bsr_patterns, make_petsc_systems
        )

        intg = self._intg

        # Keep PETSc helpers as attributes so the main build path stays local.
        self._petsc_element_solve_cls = PETScElementSolve
        self._make_bsr_patterns = make_bsr_patterns
        self._make_petsc_systems = make_petsc_systems

        # Time term and PETSc KSP controls.
        petsc_sect = 'solver-petsc'
        self.a0 = a0
        self.ksp_type = self._cfg.get(petsc_sect, 'ksp', 'gmres')
        self.rtol = self._cfg.getfloat(petsc_sect, 'sub-rtol', 1e-3)
        self.atol = self._cfg.getfloat(petsc_sect, 'sub-atol', 1e-15)
        self.max_it = self._cfg.getint(petsc_sect, 'sub-iter', 30)
        self.precon = self._cfg.get(petsc_sect, 'preconditioner', 'ilu')
        self.pc_factor_levels = self._cfg.getint(
            petsc_sect, 'pc-factor-levels', 0
        )

        for ele in intg.sys.eles:
            # PETSc writes the linear solve result into ele.du.
            ele.du = ArrayBank(ele.fpts, 1)
            self._make_update_kernel(ele, make_blusgs_update)

        # Build per-element-group BSR layouts before creating PETSc objects.
        self._make_layout()

        # Scatter PETSc solution vectors back into SOA element storage.
        self._scatter_flow_solution = self._make_solution_scatter_kernels(
            0, self._nvars
        )
        if intg._is_turb:
            self._scatter_turb_solution = self._make_solution_scatter_kernels(
                self._nvars, self._tnvars
            )

        # Backend kernels fill BSR values and RHS vectors for each solve.
        self._assemble_flow = self._make_flow_assembly_kernels()
        if intg._is_turb:
            self._assemble_turb = self._make_turb_assembly_kernels()

        # Allocate reusable PETSc matrices, vectors, and KSP wrappers.
        self._make_petsc_objects()

        # Expose solve callables through the element MetaKernel interface.
        self._bind_element_solvers()

    def _make_update_kernel(self, ele, make_blusgs_update):
        intg = self._intg
        be = intg.be
        idx_u = intg._curr_idx

        update = make_blusgs_update(ele)
        ele.update = Kernel(
            *be.make_loop(ele.neles, update), ele.upts[idx_u], ele.du
        )

    def _make_layout(self):
        intg = self._intg

        ele0 = next(iter(intg.sys.eles))
        self._nvars = nvars = ele0.nfvars
        if intg._is_turb:
            self._tnvars = tnv = ele0.nturbvars

        for ele in intg.sys.eles:
            if ele.nfvars != nvars:
                raise ValueError(
                    "petsc requires a consistent number of flow "
                    "variables across element types"
                )

            if intg._is_turb and ele.nturbvars != tnv:
                raise ValueError(
                    "petsc requires a consistent number of "
                    "turbulent variables across element types"
                )

        # Flow equations use one BSR system per element group.
        self._flow_patterns = self._make_bsr_patterns(intg.sys.eles, nvars)

        # Turbulence equations are solved separately with their own block size.
        if intg._is_turb:
            self._turb_patterns = self._make_bsr_patterns(
                intg.sys.eles, tnv
            )

    def _make_petsc_objects(self):
        try:
            from petsc4py import PETSc
        except ImportError as exc:
            raise RuntimeError(
                "petsc requires petsc4py to be installed"
            ) from exc

        self._petsc_insert_mode = PETSc.InsertMode.INSERT_VALUES

        # Use COMM_SELF so each MPI rank solves only its local element systems.
        self._flow_systems = self._make_petsc_systems(
            self._flow_patterns, PETSc
        )

        # Turbulence equations get separate PETSc systems from flow equations.
        if self._intg._is_turb:
            self._turb_systems = self._make_petsc_systems(
                self._turb_patterns, PETSc
            )

    def _bind_element_solvers(self):
        for ele in self._intg.sys.eles:
            # Bind the flow linear solve to this element group.
            ele.petsc_flow_solve = self._petsc_element_solve_cls(
                self._flow_systems[ele],
                self._assemble_flow[ele],
                self._scatter_flow_solution[ele],
                self._make_ksp,
                self._petsc_insert_mode
            )

            if self._intg._is_turb:
                # Bind the turbulence linear solve when turbulence is enabled.
                ele.petsc_turb_solve = self._petsc_element_solve_cls(
                    self._turb_systems[ele],
                    self._assemble_turb[ele],
                    self._scatter_turb_solution[ele],
                    self._make_ksp,
                    self._petsc_insert_mode
                )

    def _make_solution_scatter_kernels(self, var0, nvars):
        intg = self._intg
        be = intg.be
        kernels = {}

        # PETSc vectors are cell-major; ele.du is SOA by variable.
        for ele in intg.sys.eles:
            def copy_solution(i_begin, i_end, du, sol):
                for idx in range(i_begin, i_end):
                    base = idx*nvars

                    for kdx in range(nvars):
                        du[var0 + kdx, idx] = sol[base + kdx]

            kernels[ele] = Kernel(
                *be.make_loop(ele.neles, copy_solution), ele.du
            )

        return kernels

    def _make_flow_assembly_kernels(self):
        intg = self._intg
        be = intg.be
        idx_rhs = intg._rhs_idx
        nvars = self._nvars
        a0 = self.a0
        kernels = {}

        for ele in intg.sys.eles:
            nface = ele.nface
            pattern = self._flow_patterns[ele]
            diag_slots = pattern.diag_slots
            off_slots = pattern.off_slots
            bs2 = nvars*nvars

            def make_assemble_bsr(nface):
                # Bind nface per element group. Without this factory, mixed
                # meshes can accidentally use the last group's face count.
                def assemble_bsr(i_begin, i_end, rhs, dt, jmat, fnorm_vol,
                                 nei_ele, diag_slots, off_slots, av, bv):
                    for idx in range(i_begin, i_end):
                        base = idx*nvars

                        for kdx in range(nvars):
                            bv[base + kdx] = rhs[kdx, idx]

                        for row in range(nvars):
                            for col in range(nvars):
                                val = 0.0
                                entry = row*nvars + col

                                for jdx in range(nface):
                                    val += (
                                        jmat[0, row, col, jdx, idx]
                                        * fnorm_vol[jdx, idx]
                                    )

                                if row == col:
                                    val += 1/dt[idx] + a0

                                av[diag_slots[idx]*bs2 + entry] += val

                        for jdx in range(nface):
                            neib = nei_ele[jdx, idx]

                            if neib == idx:
                                continue

                            fv = fnorm_vol[jdx, idx]

                            for row in range(nvars):
                                for col in range(nvars):
                                    entry = row*nvars + col
                                    av[off_slots[jdx, idx]*bs2 + entry] += (
                                        jmat[1, row, col, jdx, idx]*fv
                                    )

                return assemble_bsr

            kernels[ele] = Kernel(
                *be.make_loop(ele.neles, make_assemble_bsr(nface)),
                ele.upts[idx_rhs], ele.dt, ele.jmat, ele.fnorm_vol,
                ele.nei_ele, diag_slots, off_slots
            )

        return kernels

    def _make_turb_assembly_kernels(self):
        intg = self._intg
        be = intg.be
        idx_rhs = intg._rhs_idx
        idx_u = intg._curr_idx
        nvars = self._tnvars
        a0 = self.a0
        tcfl_fac = intg._tcfl_fac
        kernels = {}

        for ele in intg.sys.eles:
            nface = ele.nface
            nfvars = ele.nfvars
            pattern = self._turb_patterns[ele]
            diag_slots = pattern.diag_slots
            off_slots = pattern.off_slots
            srcjacobian = ele.make_source_jacobian()
            array = be.local()
            bs2 = nvars*nvars

            def make_assemble_tbsr(nface, nfvars, srcjacobian):
                # Bind element-specific values for mixed meshes. In particular
                # nfvars and source Jacobian can differ from the last group.
                def assemble_tbsr(i_begin, i_end, rhs, upts, dt, tjmat,
                                  fnorm_vol, nei_ele, dsrc, diag_slots,
                                  off_slots, av, bv):
                    for idx in range(i_begin, i_end):
                        base = idx*nvars

                        for kdx in range(nvars):
                            bv[base + kdx] = rhs[nfvars + kdx, idx]

                        tmat = array((nvars, nvars), np.float64)

                        for row in range(nvars):
                            for col in range(nvars):
                                val = 0.0

                                for jdx in range(nface):
                                    val += (
                                        tjmat[0, row, col, jdx, idx]
                                        * fnorm_vol[jdx, idx]
                                    )

                                tmat[row, col] = val

                        srcjacobian(upts[:, idx], tmat, dsrc[:, idx])

                        for row in range(nvars):
                            for col in range(nvars):
                                val = tmat[row, col]
                                entry = row*nvars + col

                                if row == col:
                                    val += 1/(dt[idx]*tcfl_fac) + a0

                                av[diag_slots[idx]*bs2 + entry] += val

                        for jdx in range(nface):
                            neib = nei_ele[jdx, idx]

                            if neib == idx:
                                continue

                            fv = fnorm_vol[jdx, idx]

                            for row in range(nvars):
                                for col in range(nvars):
                                    entry = row*nvars + col
                                    av[off_slots[jdx, idx]*bs2 + entry] += (
                                        tjmat[1, row, col, jdx, idx]*fv
                                    )

                return assemble_tbsr

            kernels[ele] = Kernel(
                *be.make_loop(
                    ele.neles,
                    make_assemble_tbsr(nface, nfvars, srcjacobian)
                ),
                ele.upts[idx_rhs], ele.upts[idx_u], ele.dt, ele.tjmat,
                ele.fnorm_vol, ele.nei_ele, ele.dsrc, diag_slots,
                off_slots
            )

        return kernels

    def _make_ksp(self, A):
        from petsc4py import PETSc

        # Create a rank-local Krylov solver for one element group's matrix.
        ksp = PETSc.KSP().create(PETSc.COMM_SELF)
        ksp.setOperators(A)
        ksp.setType(self.ksp_type)
        ksp.setTolerances(
            rtol=self.rtol, atol=self.atol, max_it=self.max_it
        )

        # The preconditioner type is configurable; ILU is the default.
        pc = ksp.getPC()
        pc.setType(self.precon)
        pc.setFactorLevels(self.pc_factor_levels)

        # Let command-line PETSc options override the defaults above.
        ksp.setFromOptions()

        return ksp

    def _reduce_solver_stats(self, stats):
        nit = max(s[0] for s in stats)
        subres = max(s[1] for s in stats)

        return nit, subres

    def step(self, **kwargs):
        intg = self._intg

        resid = intg.rhs_resid(0, 1, **kwargs)

        intg.sys.eles.du.set(0)

        nit, subres = self._reduce_solver_stats(
            intg.sys.eles.petsc_flow_solve()
        )

        if intg._is_turb:
            tnit, tsubres = self._reduce_solver_stats(
                intg.sys.eles.petsc_turb_solve()
            )
            nit = max(nit, tnit)
            subres = max(subres, tsubres)

        intg.sys.eles.update()
        intg.sys.post(0)

        intg.subitnum = nit
        intg.subres = subres

        return 0, resid
