# -*- coding: utf-8 -*-
from pybaram.backends.types import ArrayBank, Kernel, MetaKernel
from pybaram.utils.misc import subclass_by_name


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
        self.subtol = self._cfg.getfloat(self._sect, 'sub-tol', 0.1)

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
        subresid = 0.0

        # Block LU-SGS subiterations.
        for it in range(self.subiter):
            # Block LU-SGS sweep
            intg.sys.eles.blusgs_sweep()

            # Compute sub-residual from all elements
            intg.sys.eles.subresid()
            drho = intg.sys.reduce_residual()[intg._res_idx]

            # Check sub-convergence
            if it == 0:
                drho1 = drho
            else:
                subresid = drho/drho1
                if subresid < self.subtol:
                    break

        intg.sys.eles.update()
        intg.sys.post(0)
        intg.subitnum = it
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
