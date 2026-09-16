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
    rank_layout_req = None

    def __init__(self, intg, cfg, sect):
        intg.impl_op = self.impl_op
        intg.rank_layout_req = self.rank_layout_req
        intg.rank_layout_name = self.name
        self._intg = intg
        self._cfg = cfg

        # Configuration section for options owned by the relaxation solver.
        self._sect = sect

    def set_a0(self, a0):
        self.a0 = a0


class BaseLUSGSRelaxation(BaseRelaxation):
    impl_op = 'spectral-radius'

    def step(self, **kwargs):
        intg = self._intg

        resid = intg.rhs_resid(0, 1, **kwargs)

        self.pack(self.a0)
        self.lusgs()
        self.update()
        intg.sys.post(0)

        return 0, resid


class LUSGSRelaxation(BaseLUSGSRelaxation):
    name = 'lu-sgs'
    rank_layout_req = 'rank-order'

    def build(self, a0, kappa=1.01):
        from pybaram.integrators.lusgs import (
            make_rank_lusgs_common, make_rank_lusgs_sweep,
            make_rank_lusgs_update, make_lusgs_pack
        )

        intg = self._intg
        be = intg.be
        idx_u = intg._curr_idx
        idx_du = intg._rhs_idx
        eles = list(intg.sys.eles)
        self.a0 = a0

        nvars = eles[0].nvars
        sys = intg.sys
        cell_ids = sys.rank_cell_ids
        nlocal = sys.rank_neles

        # Rank-cell work arrays used by pack, sweep, and update kernels.
        rank_u = be.alloc_array((nvars, nlocal))
        rank_du = be.alloc_array((nvars, nlocal))
        rank_diag = be.alloc_array((nvars, nlocal))

        pack_kernels = []
        update_kernels = []

        for ele in eles:
            ele_cell_ids = be.convert_array(cell_ids[ele])

            # Pack element-local solution data into rank-cell storage.
            pack = make_lusgs_pack(
                ele, turb_factor=getattr(intg, '_tcfl_fac', 1.0)
            )
            pack_kernels.append(Kernel(
                *be.make_loop(ele.neles, pack, ele_cell_ids),
                ele.upts[idx_u], ele.upts[idx_du], ele.dt, ele.dsrc,
                rank_u, rank_du, rank_diag
            ))

            # Scatter the rank-cell correction back to element-local storage.
            update = make_rank_lusgs_update(ele)
            update_kernels.append(Kernel(
                *be.make_loop(ele.neles, update, ele_cell_ids),
                ele.upts[idx_u], rank_du
            ))

        pre_args = (
            sys.rank_face_indptr, sys.rank_face_slots,
            sys.rank_face_area, sys.rank_rcp_vol
        )
        sweep_args = (
            sys.rank_face_indptr, sys.rank_face_slots,
            sys.rank_face_sides, sys.rank_face_neighbors,
            sys.rank_face_area, sys.rank_face_normal,
            sys.rank_rcp_vol
        )

        def make_kernels(nv, flux, lambdaf):
            # Build the diagonal preparation plus lower/upper rank sweeps.
            pre = make_rank_lusgs_common(nv, kappa=kappa)
            lower, upper = make_rank_lusgs_sweep(
                be, nvars, nv, flux, kappa=kappa
            )

            return [
                Kernel(
                    *be.make_loop(nlocal, pre, *pre_args),
                    rank_diag, lambdaf
                ),
                Kernel(
                    *be.make_loop(nlocal, lower, *sweep_args),
                    rank_u, rank_du, rank_diag, lambdaf
                ),
                Kernel(
                    *be.make_loop(nlocal, upper, *sweep_args),
                    rank_u, rank_du, rank_diag, lambdaf
                )
            ]

        # Flow variables use the shared rank face spectral radius.
        kernels = make_kernels(
            (0, eles[0].nfvars), eles[0].flux_container(),
            sys.rank_fspr
        )
        if intg._is_turb:
            # Turbulence variables use their own spectral radius storage.
            kernels += make_kernels(
                (eles[0].nfvars, nvars), eles[0].tflux_container(),
                sys.rank_tfspr
            )

        self.rank_u = rank_u
        self.rank_du = rank_du
        self.rank_diag = rank_diag
        self.pack = MetaKernel(pack_kernels)
        self.lusgs = MetaKernel(kernels)
        self.update = MetaKernel(update_kernels)


class ColoredLUSGSRelaxation(BaseLUSGSRelaxation):
    name = 'colored-lu-sgs'
    rank_layout_req = 'rank-coloring'

    def build(self, a0, kappa=1.01):
        from pybaram.integrators.lusgs import (
            make_ele_colored_lusgs_sweep, make_ele_lusgs_common,
            make_rank_lusgs_update, make_lusgs_pack
        )

        intg = self._intg
        be = intg.be
        idx_u = intg._curr_idx
        idx_du = intg._rhs_idx
        sys = intg.sys
        eles = list(sys.eles)
        self.a0 = a0

        nvars = eles[0].nvars
        nlocal = sys.rank_neles

        # Rank-cell work arrays used by pack, colored sweep, and update
        # kernels.
        rank_u = be.alloc_array((nvars, nlocal))
        rank_du = be.alloc_array((nvars, nlocal))
        rank_diag = be.alloc_array((nvars, nlocal))

        pack_kernels = []
        update_kernels = []
        for ele in eles:
            cell_ids = sys.rank_ele_cell_ids[ele]

            # Pack element-local solution data into rank-cell storage.
            pack = make_lusgs_pack(
                ele, turb_factor=getattr(intg, '_tcfl_fac', 1.0)
            )
            pack_kernels.append(Kernel(
                *be.make_loop(ele.neles, pack, cell_ids),
                ele.upts[idx_u], ele.upts[idx_du], ele.dt, ele.dsrc,
                rank_u, rank_du, rank_diag
            ))

            # Scatter the rank-cell correction back to element-local storage.
            update = make_rank_lusgs_update(ele)
            update_kernels.append(Kernel(
                *be.make_loop(ele.neles, update, cell_ids),
                ele.upts[idx_u], rank_du
            ))

        def make_kernels(nv, flux, lambdaf):
            # Build element-local diagonal preparation kernels and colored
            # lower/upper sweep kernels.
            wave_factors = {
                ele: be.alloc_array((ele.nface, ele.neles))
                for ele in eles
            }
            pre_kernels = []
            sweeps = {}
            for ele in eles:
                pre = make_ele_lusgs_common(
                    ele, nv, kappa=kappa
                )
                pre_kernels.append(Kernel(
                    *be.make_loop(
                        ele.neles, pre,
                        sys.rank_ele_cell_ids[ele],
                        sys.rank_ele_face_refs[ele],
                        sys.rank_ele_face_factors[ele]
                    ),
                    rank_diag, wave_factors[ele], lambdaf
                ))
                lower, upper = make_ele_colored_lusgs_sweep(
                    be, ele, nvars, nv, flux
                )
                sweeps[ele] = {'lower': lower, 'upper': upper}

            def make_color_kernels(color, direction):
                kernels = []
                for ele in eles:
                    # Offsets select this rank color inside
                    # rank_ele_color_order[ele].
                    offsets = sys.rank_ele_color_offsets[ele]
                    begin, end = offsets[color:color + 2]
                    if begin == end:
                        continue

                    if direction == 'lower':
                        neighbors = sys.rank_ele_lower_neighbors[ele]
                    else:
                        neighbors = sys.rank_ele_upper_neighbors[ele]

                    sweep_args = (
                        sys.rank_ele_color_order[ele],
                        sys.rank_ele_cell_ids[ele], neighbors,
                        sys.rank_ele_face_factors[ele],
                        sys.rank_ele_face_normals[ele],
                        wave_factors[ele]
                    )
                    kernels.append(Kernel(
                        *be.make_loop(
                            end, sweeps[ele][direction],
                            *sweep_args, n0=begin
                        ),
                        rank_u, rank_du, rank_diag
                    ))
                return kernels

            ncolors = sys.rank_ncolors
            # Lower sweeps advance colors; upper sweeps walk them backward.
            lower_kernels = [
                kern
                for color in range(ncolors)
                for kern in make_color_kernels(color, 'lower')
            ]
            upper_kernels = [
                kern
                for color in range(ncolors - 1, -1, -1)
                for kern in make_color_kernels(color, 'upper')
            ]

            return [
                *pre_kernels, *lower_kernels, *upper_kernels
            ], wave_factors

        # Flow variables use the shared rank face spectral radius.
        flow_kernels, flow_wave_factors = make_kernels(
            (0, eles[0].nfvars), eles[0].flux_container(),
            sys.rank_fspr
        )
        kernels = flow_kernels
        if intg._is_turb:
            # Turbulence variables use their own spectral radius storage.
            turb_kernels, turb_wave_factors = make_kernels(
                (eles[0].nfvars, nvars), eles[0].tflux_container(),
                sys.rank_tfspr
            )
            kernels += turb_kernels

        self.rank_u = rank_u
        self.rank_du = rank_du
        self.rank_diag = rank_diag
        self.flow_wave_factors = flow_wave_factors
        if intg._is_turb:
            self.turb_wave_factors = turb_wave_factors
        self.pack = MetaKernel(pack_kernels)
        self.lusgs = MetaKernel(kernels)
        self.update = MetaKernel(update_kernels)


class BaseBlockLUSGSRelaxation(BaseRelaxation):
    impl_op = 'approx-jacobian'

    def _init_subiteration_controls(self):
        # Block LU-SGS subiteration controls.
        self.subiter = self._cfg.getint(self._sect, 'sub-iter', 10)
        self.subrtol = self._cfg.getfloat(self._sect, 'sub-rtol', 0.1)
        self.subatol = self._cfg.getfloat(self._sect, 'sub-atol', 0.0)

    def step(self, **kwargs):
        intg = self._intg

        resid = intg.rhs_resid(0, 1, **kwargs)

        self.pack(self.a0)

        # Reset correction histories.
        self.reset()

        # Compute diagonal matrix
        self.pre_blusgs()
        subresid = 1.0

        # Block LU-SGS subiterations.
        for it in range(self.subiter):
            # Block LU-SGS sweep
            self.blusgs_sweep()

            # Compute sub-residual from all elements
            self.subresid()
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

        self.update()
        intg.sys.post(0)
        intg.subitnum = it + 1
        intg.subres = subresid

        return 0, resid


class BlockLUSGSRelaxation(BaseBlockLUSGSRelaxation):
    name = 'blu-sgs'
    rank_layout_req = 'rank-order'

    def build(self, a0):
        from pybaram.integrators.blusgs import (
            make_rank_blusgs_pack, make_rank_blusgs_sweep,
            make_rank_blusgs_update, make_rank_pre_blusgs,
            make_rank_sub_residual, make_rank_tblusgs_pack
        )

        intg = self._intg
        be = intg.be

        self._init_subiteration_controls()
        self.a0 = a0
        idx_u = intg._curr_idx
        idx_rhs = intg._rhs_idx
        sys = intg.sys
        eles = list(sys.eles)
        nlocal = sys.rank_neles
        nvars = eles[0].nvars
        nfvars = eles[0].nfvars

        # Rank-cell work arrays for block RHS, correction, and subiteration
        # residual history.
        rank_rhs = be.alloc_array((nvars, nlocal))
        rank_du = be.alloc_array((nvars, nlocal))
        rank_dup = be.alloc_array((nvars, nlocal))

        # Flow diagonal blocks are stored in LU-factorized form after pre.
        flow_diag = be.alloc_array((nfvars, nfvars, nlocal))

        pack_kernels = []
        update_kernels = []
        subresid_kernels = []
        for ele in eles:
            cell_ids = be.convert_array(sys.rank_cell_ids[ele])

            # Pack flow RHS and initialize flow diagonal blocks.
            pack = make_rank_blusgs_pack(ele)
            pack_kernels.append(Kernel(
                *be.make_loop(ele.neles, pack, cell_ids),
                ele.upts[idx_rhs], ele.dt, rank_rhs, flow_diag
            ))

            # Scatter the rank-cell correction back to element-local storage.
            update = make_rank_blusgs_update(ele)
            update_kernels.append(Kernel(
                *be.make_loop(ele.neles, update, cell_ids),
                ele.upts[idx_u], rank_du
            ))

            # Track correction changes for block LU-SGS sub-convergence.
            subresid = make_rank_sub_residual(ele)
            subresid_kernels.append(Kernel(
                *be.make_loop(ele.neles, subresid, cell_ids),
                ele.vol, rank_du, rank_dup, ele.resid_out
            ))

        face_args = (
            sys.rank_face_indptr, sys.rank_face_slots,
            sys.rank_face_sides, sys.rank_face_neighbors,
            sys.rank_face_area, sys.rank_rcp_vol
        )
        pre_face_args = (
            sys.rank_face_indptr, sys.rank_face_slots,
            sys.rank_face_sides, sys.rank_face_area,
            sys.rank_rcp_vol
        )

        # Assemble and factorize local diagonal blocks from face Jacobians.
        pre_flow = make_rank_pre_blusgs(be, nfvars)
        pre_kernels = [Kernel(
            *be.make_loop(nlocal, pre_flow, *pre_face_args),
            flow_diag, sys.rank_jmat
        )]

        # Flow lower/upper sweeps use the rank-order CSR face layout.
        lower, upper = make_rank_blusgs_sweep(be, 0, nfvars)
        sweep_kernels = [
            Kernel(
                *be.make_loop(nlocal, lower, *face_args),
                rank_rhs, rank_du, flow_diag, sys.rank_jmat
            ),
            Kernel(
                *be.make_loop(nlocal, upper, *face_args),
                rank_rhs, rank_du, flow_diag, sys.rank_jmat
            )
        ]

        if intg._is_turb:
            nturbvars = nvars - nfvars

            # Turbulence uses a separate block diagonal and source Jacobian.
            turb_diag = be.alloc_array((nturbvars, nturbvars, nlocal))
            for ele in eles:
                cell_ids = be.convert_array(sys.rank_cell_ids[ele])
                pack_turb = make_rank_tblusgs_pack(
                    ele, ele.make_source_jacobian(),
                    factor=intg._tcfl_fac
                )
                pack_kernels.append(Kernel(
                    *be.make_loop(ele.neles, pack_turb, cell_ids),
                    ele.upts[idx_u], ele.dt, ele.dsrc, turb_diag
                ))

            # Assemble and factorize turbulence diagonal blocks.
            pre_turb = make_rank_pre_blusgs(
                be, nturbvars
            )
            pre_kernels.append(Kernel(
                *be.make_loop(nlocal, pre_turb, *pre_face_args),
                turb_diag, sys.rank_tjmat
            ))

            # Turbulence sweeps operate on the turbulence variable block.
            tlower, tupper = make_rank_blusgs_sweep(
                be, nfvars, nturbvars
            )
            sweep_kernels += [
                Kernel(
                    *be.make_loop(nlocal, tlower, *face_args),
                    rank_rhs, rank_du, turb_diag, sys.rank_tjmat
                ),
                Kernel(
                    *be.make_loop(nlocal, tupper, *face_args),
                    rank_rhs, rank_du, turb_diag, sys.rank_tjmat
                )
            ]

        def reset():
            # Start each block LU-SGS solve from a zero correction.
            rank_du[:] = 0
            rank_dup[:] = 0

        # Keep reusable work arrays and kernel groups for step().
        self.rank_rhs = rank_rhs
        self.rank_du = rank_du
        self.rank_dup = rank_dup
        self.flow_diag = flow_diag
        if intg._is_turb:
            self.turb_diag = turb_diag
        self.pack = MetaKernel(pack_kernels)
        self.reset = reset
        self.pre_blusgs = MetaKernel(pre_kernels)
        self.blusgs_sweep = MetaKernel(sweep_kernels)
        self.subresid = MetaKernel(subresid_kernels)
        self.update = MetaKernel(update_kernels)


class ColoredBlockLUSGSRelaxation(BaseBlockLUSGSRelaxation):
    name = 'colored-blu-sgs'
    rank_layout_req = 'rank-coloring'

    def build(self, a0):
        from pybaram.integrators.blusgs import (
            make_ele_colored_blusgs_sweep, make_ele_pre_blusgs,
            make_rank_blusgs_diag_pack, make_rank_blusgs_update,
            make_rank_sub_residual, make_rank_tblusgs_pack
        )

        intg = self._intg
        be = intg.be

        self._init_subiteration_controls()
        self.a0 = a0
        idx_u = intg._curr_idx
        idx_rhs = intg._rhs_idx
        sys = intg.sys
        eles = list(sys.eles)

        nlocal = sys.rank_neles
        nvars = eles[0].nvars
        nfvars = eles[0].nfvars

        # Rank-cell correction arrays and flow diagonal blocks.
        rank_du = be.alloc_array((nvars, nlocal))
        rank_dup = be.alloc_array((nvars, nlocal))
        flow_diag = be.alloc_array((nfvars, nfvars, nlocal))

        pack_kernels = []
        update_kernels = []
        subresid_kernels = []
        for ele in eles:
            cell_ids = sys.rank_ele_cell_ids[ele]

            # Initialize flow diagonal blocks in rank-cell storage.
            pack = make_rank_blusgs_diag_pack(ele)
            pack_kernels.append(Kernel(
                *be.make_loop(ele.neles, pack, cell_ids),
                ele.dt, flow_diag
            ))

            # Scatter the rank-cell correction back to element-local storage.
            update = make_rank_blusgs_update(ele)
            update_kernels.append(Kernel(
                *be.make_loop(ele.neles, update, cell_ids),
                ele.upts[idx_u], rank_du
            ))

            # Track correction changes for block LU-SGS sub-convergence.
            subresid = make_rank_sub_residual(ele)
            subresid_kernels.append(Kernel(
                *be.make_loop(ele.neles, subresid, cell_ids),
                ele.vol, rank_du, rank_dup, ele.resid_out
            ))

        # Element-local off-diagonal blocks are indexed by face and cell.
        flow_offdiag = {
            ele: be.alloc_array(
                (nfvars, nfvars, ele.nface, ele.neles)
            )
            for ele in eles
        }
        pre_kernels = []
        for ele in eles:
            # Assemble and factorize flow diagonal blocks; cache off-diagonal
            # face blocks for colored sweeps.
            pre_flow = make_ele_pre_blusgs(be, ele, nfvars)
            pre_kernels.append(Kernel(
                *be.make_loop(
                    ele.neles, pre_flow,
                    sys.rank_ele_cell_ids[ele],
                    sys.rank_ele_face_refs[ele],
                    sys.rank_ele_face_factors[ele]
                ),
                flow_diag, flow_offdiag[ele], sys.rank_jmat
            ))

        def make_sweeps(var0, nblock, diag, offdiag):
            # Build one element-local colored sweep kernel per element type.
            sweeps = {
                ele: make_ele_colored_blusgs_sweep(
                    be, ele, var0, nblock
                )
                for ele in eles
            }

            def make_color_kernels(color):
                kernels = []
                for ele in eles:
                    # Offsets select this rank color inside
                    # rank_ele_color_order[ele].
                    offsets = sys.rank_ele_color_offsets[ele]
                    begin, end = offsets[color:color + 2]
                    if begin == end:
                        continue

                    face_args = (
                        sys.rank_ele_color_order[ele],
                        sys.rank_ele_cell_ids[ele],
                        sys.rank_ele_neighbors[ele]
                    )
                    kernels.append(Kernel(
                        *be.make_loop(
                            end, sweeps[ele], *face_args, n0=begin
                        ),
                        ele.upts[idx_rhs], rank_du, diag, offdiag[ele]
                    ))
                return kernels

            ncolors = sys.rank_ncolors
            # Forward and backward passes walk color barriers in opposite
            # directions.
            forward = [
                kern
                for color in range(ncolors)
                for kern in make_color_kernels(color)
            ]
            backward = [
                kern
                for color in range(ncolors - 1, -1, -1)
                for kern in make_color_kernels(color)
            ]
            return [*forward, *backward]

        sweep_kernels = make_sweeps(
            0, nfvars, flow_diag, flow_offdiag
        )

        if intg._is_turb:
            nturbvars = nvars - nfvars

            # Turbulence uses a separate block diagonal and source Jacobian.
            turb_diag = be.alloc_array((nturbvars, nturbvars, nlocal))
            for ele in eles:
                cell_ids = sys.rank_ele_cell_ids[ele]
                pack_turb = make_rank_tblusgs_pack(
                    ele, ele.make_source_jacobian(),
                    factor=intg._tcfl_fac
                )
                pack_kernels.append(Kernel(
                    *be.make_loop(ele.neles, pack_turb, cell_ids),
                    ele.upts[idx_u], ele.dt, ele.dsrc, turb_diag
                ))

            # Turbulence off-diagonal blocks mirror the flow layout with a
            # different block size.
            turb_offdiag = {
                ele: be.alloc_array(
                    (nturbvars, nturbvars, ele.nface, ele.neles)
                )
                for ele in eles
            }
            for ele in eles:
                # Assemble and factorize turbulence diagonal blocks.
                pre_turb = make_ele_pre_blusgs(
                    be, ele, nturbvars
                )
                pre_kernels.append(Kernel(
                    *be.make_loop(
                        ele.neles, pre_turb,
                        sys.rank_ele_cell_ids[ele],
                        sys.rank_ele_face_refs[ele],
                        sys.rank_ele_face_factors[ele]
                    ),
                    turb_diag, turb_offdiag[ele], sys.rank_tjmat
                ))
            sweep_kernels += make_sweeps(
                nfvars, nturbvars, turb_diag, turb_offdiag
            )

        def reset():
            # Start each block LU-SGS solve from a zero correction.
            rank_du[:] = 0
            rank_dup[:] = 0

        # Keep reusable work arrays and kernel groups for step().
        self.rank_du = rank_du
        self.rank_dup = rank_dup
        self.flow_diag = flow_diag
        self.flow_offdiag = flow_offdiag
        if intg._is_turb:
            self.turb_diag = turb_diag
            self.turb_offdiag = turb_offdiag
        self.pack = MetaKernel(pack_kernels)
        self.reset = reset
        self.pre_blusgs = MetaKernel(pre_kernels)
        self.blusgs_sweep = MetaKernel(sweep_kernels)
        self.subresid = MetaKernel(subresid_kernels)
        self.update = MetaKernel(update_kernels)


class _BasePETScRelaxation(BaseRelaxation):
    """Shared PETSc KSP relaxation build and step logic."""

    impl_op = 'approx-jacobian'
    rank_layout_req = 'rank-order'

    def _set_ksp_divergence_policy(self):
        # Configure how PETSc KSP divergence is reported after each solve.
        policy = self._cfg.get(
            'solver-petsc', 'ksp-divergence', 'warn'
        ).lower()
        if policy not in ('ignore', 'warn', 'raise'):
            raise ValueError(
                "solver-petsc ksp-divergence must be ignore, warn, or raise"
            )
        self._ksp_divergence_policy = policy

    def _check_ksp_reason(self, reason, label):
        # Apply the configured policy to PETSc's convergence reason.
        if reason >= 0 or self._ksp_divergence_policy == 'ignore':
            return

        message = "{} KSP diverged (PETSc reason {})".format(label, reason)
        if self._ksp_divergence_policy == 'raise':
            raise RuntimeError(message)
        if self._intg._comm.rank == 0:
            print("Warning: {}".format(message))

    def _build_petsc(
        self, a0, solve_cls, make_patterns, parallel, solve_kwargs=None
    ):
        from pybaram.integrators.petsc import (
            make_rank_petsc_scatter, make_rank_petsc_update
        )

        intg = self._intg
        be = intg.be
        sys = intg.sys
        idx_u = intg._curr_idx
        if solve_kwargs is None:
            solve_kwargs = {}

        # Read KSP divergence handling before PETSc solve objects are built.
        self._set_ksp_divergence_policy()

        # Store the time term and KSP controls for linear solves.
        self.a0 = a0
        petsc_sect = 'solver-petsc'
        ksp_opts = {
            'atol': self._cfg.getfloat(petsc_sect, 'sub-atol', 1e-15),
            'rtol': self._cfg.getfloat(petsc_sect, 'sub-rtol', 1e-3),
            'max_it': self._cfg.getint(petsc_sect, 'sub-iter', 30),
            'type': self._cfg.get(petsc_sect, 'ksp', 'gmres'),
            'pc': self._cfg.get(petsc_sect, 'preconditioner', 'ilu'),
            'pc_levels': self._cfg.getint(
                petsc_sect, 'pc-factor-levels', 0
            ),
            'parallel': parallel
        }

        for ele in intg.sys.eles:
            # PETSc solutions are scattered into ele.du before update().
            ele.du = ArrayBank(ele.fpts, 1)
            update = make_rank_petsc_update(ele)
            ele.update = Kernel(
                *be.make_loop(ele.neles, update), ele.upts[idx_u], ele.du
            )

        ele0 = next(iter(sys.eles))
        self._nvars = nvars = ele0.nfvars
        if intg._is_turb:
            self._tnvars = tnv = ele0.nturbvars
        else:
            tnv = None

        # BSR patterns provide PETSc rows and backend assembly slots.
        self._flow_pattern, self._turb_pattern = make_patterns(nvars, tnv)

        # Element-local cells are indexed into rank-cell PETSc vectors.
        self._cell_ids = {
            ele: be.convert_array(
                np.asarray(sys.rank_cell_ids[ele], dtype=np.int32)
            )
            for ele in sys.eles
        }

        def make_scatter(var0, nv):
            kernels = []
            for ele in intg.sys.eles:
                # Scatter rank-cell-major vectors back to element SOA storage.
                scatter = make_rank_petsc_scatter(var0, nv)
                kernels.append(Kernel(
                    *be.make_loop(ele.neles, scatter, self._cell_ids[ele]),
                    ele.du
                ))

            return MetaKernel(kernels)

        self._scatter_flow_solution = make_scatter(0, nvars)
        if intg._is_turb:
            self._scatter_turb_solution = make_scatter(nvars, tnv)

        # Pack RHS/diagonal terms, then add face Jacobian blocks.
        self._assemble_flow = self._make_flow_assembly_kernels()
        if intg._is_turb:
            self._assemble_turb = self._make_turb_assembly_kernels()

        # Create PETSc solve objects; their matrices, vectors, and KSPs are
        # reused between steps.
        self.petsc_flow_solve = solve_cls(
            self._flow_pattern, self._assemble_flow,
            self._scatter_flow_solution, opts=ksp_opts, **solve_kwargs
        )
        if intg._is_turb:
            self.petsc_turb_solve = solve_cls(
                self._turb_pattern, self._assemble_turb,
                self._scatter_turb_solution, opts=ksp_opts, **solve_kwargs
            )

    def _make_assembly_kernel(self, pattern, nvars, jmat, pack_kernels):
        from pybaram.integrators.petsc import make_rank_petsc_face_assemble

        intg = self._intg
        be = intg.be
        sys = intg.sys

        # Add face Jacobian blocks into the PETSc BSR value array.
        face_assemble = make_rank_petsc_face_assemble(nvars)
        face_kernel = Kernel(
            *be.make_loop(
                sys.rank_neles, face_assemble,
                sys.rank_face_indptr, sys.rank_face_slots,
                sys.rank_face_sides, sys.rank_face_area, sys.rank_rcp_vol,
                pattern.diag_slots, pattern.off_slots, jmat
            )
        )

        # Run all element pack kernels for RHS and diagonal BSR terms.
        pack = MetaKernel(pack_kernels)

        def assemble(values, rhs):
            # First pack element terms, then add face Jacobian blocks.
            pack(self.a0, values, rhs)
            face_kernel(values)

        return assemble

    def _make_flow_assembly_kernels(self):
        from pybaram.integrators.petsc import make_rank_petsc_rhs_pack

        intg = self._intg
        be = intg.be
        idx_rhs = intg._rhs_idx
        nvars = self._nvars
        sys = intg.sys
        pattern = self._flow_pattern
        pack_kernels = []

        for ele in intg.sys.eles:
            # Flow RHS and pseudo-time diagonal are packed per rank cell.
            pack = make_rank_petsc_rhs_pack(0, nvars)
            pack_kernels.append(Kernel(
                *be.make_loop(ele.neles, pack, self._cell_ids[ele]),
                ele.upts[idx_rhs], ele.dt, pattern.diag_slots
            ))

        # Combine flow packing with flow face Jacobian assembly.
        return self._make_assembly_kernel(
            pattern, nvars, sys.rank_jmat, pack_kernels
        )

    def _make_turb_assembly_kernels(self):
        from pybaram.integrators.petsc import make_rank_petsc_turb_pack

        intg = self._intg
        be = intg.be
        idx_rhs = intg._rhs_idx
        idx_u = intg._curr_idx
        nvars = self._tnvars
        tcfl_fac = intg._tcfl_fac
        sys = intg.sys
        pattern = self._turb_pattern
        pack_kernels = []

        for ele in intg.sys.eles:
            # Turbulence packing also adds the local source Jacobian.
            srcjacobian = ele.make_source_jacobian()
            pack = make_rank_petsc_turb_pack(
                be, ele, srcjacobian, factor=tcfl_fac
            )
            pack_kernels.append(Kernel(
                *be.make_loop(ele.neles, pack, self._cell_ids[ele]),
                ele.upts[idx_rhs], ele.upts[idx_u], ele.dt, ele.dsrc,
                pattern.diag_slots
            ))

        # Combine turbulence packing with turbulence face Jacobian assembly.
        return self._make_assembly_kernel(
            pattern, nvars, sys.rank_tjmat, pack_kernels
        )

    def step(self, **kwargs):
        intg = self._intg

        # Evaluate residuals and approximate Jacobian blocks.
        resid = intg.rhs_resid(0, 1, **kwargs)

        # Clear element correction buffers before PETSc scatters solve results.
        intg.sys.eles.du.set(0)

        # Solve the flow correction in PETSc rank-cell storage.
        nit, subres, reason = self.petsc_flow_solve()
        self._check_ksp_reason(reason, 'flow')

        if intg._is_turb:
            # Turbulence is solved as a separate PETSc system.
            tnit, tsubres, treason = self.petsc_turb_solve()
            self._check_ksp_reason(treason, 'turbulence')
            nit = max(nit, tnit)
            subres = max(subres, tsubres)

        # Scatter corrections to elements, update solution, and postprocess.
        intg.sys.eles.update()
        intg.sys.post(0)

        # Report the worst PETSc sub-solve statistics for this step.
        intg.subitnum = nit
        intg.subres = subres

        return 0, resid


class PETScRankRelaxation(_BasePETScRelaxation):
    """Rank-local PETSc KSP relaxation."""

    name = 'petsc-rank'

    def build(self, a0):
        from pybaram.integrators.petsc import (
            PETScRankSolve, make_rank_bsr_pattern
        )

        sys = self._intg.sys

        def make_patterns(nvars, tnv):
            # Rank-local PETSc uses rank-cell block columns.
            flow_pattern = make_rank_bsr_pattern(
                sys.rank_face_indptr, sys.rank_face_neighbors,
                sys.rank_neles, nvars
            )
            turb_pattern = None
            if tnv is not None:
                turb_pattern = make_rank_bsr_pattern(
                    sys.rank_face_indptr, sys.rank_face_neighbors,
                    sys.rank_neles, tnv
                )

            return flow_pattern, turb_pattern

        self._build_petsc(
            a0, PETScRankSolve, make_patterns, parallel=False
        )


class PETScGlobalRelaxation(_BasePETScRelaxation):
    """Distributed PETSc KSP relaxation with MPI-interface couplings."""

    name = 'petsc'

    def build(self, a0):
        from pybaram.integrators.petsc import (
            BSRPattern, PETScGlobalSolve, make_global_bsr_pattern
        )

        intg = self._intg
        sys = intg.sys

        def make_patterns(nvars, tnv):
            # Distributed PETSc uses local rows and global block columns.
            flow_pattern = make_global_bsr_pattern(
                sys.rank_face_indptr, sys.rank_face_neighbors,
                sys.rank_neles, sys.mpiint, list(sys.eles),
                sys.rank_cell_ids, intg._comm, nvars
            )
            turb_pattern = None
            if tnv is not None:
                ncells = flow_pattern.global_ndof//nvars

                # Reuse connectivity with turbulence block/vector sizes.
                turb_pattern = BSRPattern(
                    rowptr=flow_pattern.rowptr,
                    colidx=flow_pattern.colidx,
                    diag_slots=flow_pattern.diag_slots,
                    off_slots=flow_pattern.off_slots,
                    local_ndof=sys.rank_neles*tnv,
                    global_ndof=ncells*tnv,
                    bsize=tnv
                )

            return flow_pattern, turb_pattern

        # Pack/assemble/scatter kernels still operate on locally owned rows.
        self._build_petsc(
            a0, PETScGlobalSolve, make_patterns,
            parallel=self._intg._comm.size > 1,
            solve_kwargs={'comm': self._intg._comm}
        )
