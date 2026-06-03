# -*- coding: utf-8 -*-
import numpy as np

from pybaram.solvers.base import BaseElements
from pybaram.solvers.euler.jacobian import make_convective_jacobian
from pybaram.backends.types import ArrayBank, Kernel, NullKernel
from pybaram.utils.np import eps


class BaseAdvecElements(BaseElements):
    def construct_kernels(self, vertex, nreg):
        self.vertex = vertex

        # Upts : Solution vector
        self.upts = upts = [self.be.alloc_array(None, src=self._ics) for i in range(nreg)]
        del(self._ics)

        # Solution vector bank and assign upts index
        self.upts_in = upts_in = ArrayBank(upts, 0)
        self.upts_out = upts_out = ArrayBank(upts, 1)
        self.upts_res = upts_res = ArrayBank(upts, 1)

        # Construct arrays for flux points, dt and derivatives of source term
        self.fpts = fpts = self.be.alloc_array((self.nface, self.nvars, self.neles))
        self.dt = self.be.alloc_array((self.neles,))

        #TODO: Maybe re-use vpts?
        self.dsrc = self.be.alloc_array((self.nvars, self.neles), init=0)

        # Residual vector
        self.h_resid, self.d_resid = self.be.alloc_array((self.nvars,), mapped=True)
        
        # Re-use fpts for residual
        self.resid_out = ArrayBank(fpts, 0)

        if self.order > 1:
            # Array for gradient and limiter
            self.grad = grad = self.be.alloc_array((self.ndims, self.nvars, self.neles), init=0)
            lim = self.be.alloc_array((self.nvars, self.neles), init=1)
            limiter = self.cfg.get('solver', 'limiter', 'none')

            # Prepare vertex array
            vpts = vertex.make_array(limiter)

        # Build kernels
        # Kernel to compute flux points
        self.compute_fpts = Kernel(*self._make_compute_fpts(), upts_in, fpts)

        # Kernel to compute divergence of solution
        self.div_upts = Kernel(*self._make_div_upts(), upts_out, fpts, upts_in)

        # Kernel to compute residuals
        self.compute_resid = Kernel(*self._make_compute_resid(), upts_res, self.resid_out)
        self.reduce_resid = Kernel(self.be.reduce_array(self.nvars), self.resid_out, self.d_resid)

        if self.order > 1:
            # Kernel to compute gradient
            self.compute_grad = Kernel(*self._make_grad(), fpts, grad)

            # Kernel for linear reconstruction
            self.compute_recon = Kernel(
                *self._make_recon(), upts_in, grad, lim, fpts)

            if limiter != 'none':
                # Kenerl to compute slope limiter (MLP-u)
                self.compute_mlp_u = Kernel(
                    *self._make_mlp_u(limiter), upts_in, grad, vpts, lim)
            else:
                self.compute_mlp_u = NullKernel
        else:
            self.compute_grad = NullKernel
            self.compute_recon = NullKernel
            self.compute_mlp_u = NullKernel

        # Kernel to post-process
        self.post = Kernel(*self._make_post(), upts_in)

    def _make_compute_resid(self):
        vol = self.vol
        nvars = self.nvars

        def _compute_resid(i_begin, i_end, vol, upts, resid_out):
            for idx in range(i_begin, i_end):
                for jdx in range(nvars):
                    resid_out[jdx, idx] = upts[jdx, idx]**2 * vol[idx]
        
        return self.be.make_loop(self.neles, _compute_resid, vol)

    def _make_compute_fpts(self):
        nvars, nface = self.nvars, self.nface

        def _compute_fpts(i_begin, i_end, upts, fpts):
            # Copy upts to fpts
            for idx in range(i_begin, i_end):
                for j in range(nvars):
                    tmp = upts[j, idx]
                    for k in range(nface):
                        fpts[k, j, idx] = tmp
        
        return self.be.make_loop(self.neles, _compute_fpts)

    def _source_exprs(self):
        # Position, constants and numerical functions
        subs = {x: 'xc[{0}, idx]'.format(i)
                for i, x in enumerate('xyz'[:self.ndims])}
        subs.update(self._const)
        subs.update({'sin': 'np.sin', 'cos': 'np.cos',
                     'exp': 'np.exp', 'tanh': 'np.tanh'})
        
        # Conservative variables
        subs.update({v.lower() : 'upts[{0}, idx]'.format(i)
                     for i, v in enumerate(self.conservars)})

        # Parse source term
        src = [self.cfg.getexpr('solver-source-terms', k, subs, default=0.0)
               for k in self.conservars]

        return src, any('xc[' in s for s in src)

    def _make_div_upts(self):
        # Global variables for compile
        rcp_vol = self.be.convert_array(self.rcp_vol)
        src, has_xc = self._source_exprs()

        # Construct function text
        if has_xc:
            args = 'rcp_vol, xc, rhs, fpts, upts'
        else:
            args = 'rcp_vol, rhs, fpts, upts'
        f_txt = (
            f"def _div_upts(i_begin, i_end, {args}, t=0):\n"
            f"    for idx in range(i_begin, i_end): \n"
            f"        rcp_voli = rcp_vol[idx]\n"
        )
        for j, s in enumerate(src):
            subtxt = "+".join("fpts[{},{},idx]".format(i, j)
                              for i in range(self.nface))
            f_txt += "        rhs[{}, idx] = -rcp_voli*({}) + {}\n".format(
                j, subtxt, s)

        # Execute python function and save in lvars
        lvars = {}
        exec(f_txt, {"np": np}, lvars)

        # Compile the function
        if has_xc:
            xc = self.be.convert_array(self.xc.T)
            return self.be.make_loop(self.neles, lvars["_div_upts"],
                                     rcp_vol, xc, src=f_txt)
        else:
            return self.be.make_loop(self.neles, lvars["_div_upts"],
                                     rcp_vol, src=f_txt)

    def _make_grad(self):
        nface, ndims, nvars = self.nface, self.ndims, self.nvars

        # Gradient operator 
        op = self.be.convert_array(self._prelsq)

        def _cal_grad(i_begin, i_end, op, fpts, grad):
            # Elementwise dot product
            # TODO: Reduce accesing global array
            for i in range(i_begin, i_end):
                for l in range(nvars):
                    for k in range(ndims):
                        tmp = 0
                        for j in range(nface):
                            tmp += op[k, j, i]*fpts[j, l, i]
                        grad[k, l, i] = tmp

        # Compile the function
        return self.be.make_loop(self.neles, _cal_grad, op)       

    def _make_recon(self):
        nface, ndims, nvars = self.nface, self.ndims, self.nvars

        # Displacement vector
        op = self.be.convert_array(self.dxf)

        def _cal_recon(i_begin, i_end, op, upts, grad, lim, fpts):
            # Elementwise dot product and scale with limiter
            # TODO: Reduce accesing global array
            for i in range(i_begin, i_end):
                for l in range(nvars):
                    for k in range(nface):
                        tmp = 0
                        for j in range(ndims):
                            tmp += op[k, j, i]*grad[j, l, i]
                        fpts[k, l, i] = upts[l, i] + lim[l, i]*tmp

        return self.be.make_loop(self.neles, _cal_recon, op)

    def _make_mlp_u(self, limiter):
        nvtx, ndims, nvars = self.nvtx, self.ndims, self.nvars

        dx = self.be.convert_array(self.dxv)
        cons = self.be.convert_array(self._vcon.T)

        def u1(dup, dum, ee2):
            # u1 function
            return min(1.0, dup/dum)

        def u2(dup, dum, ee2):
            # u2 function
            dup2 = dup**2
            dum2 = dum**2
            dupm = dup*dum
            return ((dup2 + ee2)*dum + 2*dum2*dup)/(dup2 + 2*dum2 + dupm + ee2)/dum

        # x_i^1.5 : Characteristic length for u2 function
        le32 = self.be.convert_array(self.le**1.5)

        if limiter == 'mlp-u2':
            is_u2 = True
            u2k = self.cfg.getfloat('solver', 'u2k', 5.0)

            # Don't use ee2 for very small u2k
            if u2k < eps:
                is_u2 = False

            limf = self.be.compile(u2)
        else:
            is_u2 = False
            u2k = 0.0
            limf = self.be.compile(u1)

        def _cal_mlp_u(i_begin, i_end, cons, dx, le32, upts, grad, vext, lim):
            for i in range(i_begin, i_end):
                for j in range(nvtx):
                    vi = cons[j, i]
                    for k in range(nvars):
                        duv = 0

                        if is_u2:
                            # parameter for u2 
                            dvv = vext[0, k, vi] - vext[1, k, vi]
                            ee = dvv / le32[i] / u2k
                            ee2 = u2k*dvv**2/(ee + 1.0)
                        else:
                            ee2 = 0.0

                        # Difference of values between vertex and cell-center
                        for l in range(ndims):
                            duv += dx[j, l, i]*grad[l, k, i]

                        # MLP-u slope limiter
                        if duv > eps:
                            limj = limf(
                                (vext[0, k, vi] - upts[k, i]), duv, ee2)
                        elif duv < -eps:
                            limj = limf(
                                (vext[1, k, vi] - upts[k, i]), duv, ee2)
                        else:
                            limj = 1.0

                        if j == 0:
                            lim[k, i] = limj
                        else:
                            lim[k, i] = min(lim[k, i], limj)

        return self.be.make_loop(self.neles, _cal_mlp_u, cons, dx, le32)

    def _make_post(self):
        # Get post-process function
        _fix_nonPys = self.fix_nonPys_container()

        def post(i_begin, i_end, upts):
            # Apply the function over eleemnts
            for idx in range(i_begin, i_end):
                _fix_nonPys(upts[:, idx])

        return self.be.make_loop(self.neles, post)
