from pybaram.utils.inverse import make_lu_dcmp, make_substitution

import numpy as np


def make_blusgs_update(ele):
    # Number of variables
    nvars = ele.nvars

    def _update(i_begin, i_end, uptsb, dub, dubp1):
        for idx in range(i_begin, i_end):
            for kdx in range(nvars):
                # Update solution by adding residual
                uptsb[kdx, idx] += dub[kdx, idx]

                # Initialize dub array
                dub[kdx, idx] = 0.0
            
            # Initialize dubp1 array
            dubp1[idx] = 0.0

    return _update


def make_pre_blusgs(be, ele, nv, factor=1.0):
    # Number of faces
    nface = ele.nface

    # Number of variables
    dnv = int(nv[1] - nv[0])

    # LU decomposition function
    dcmp_func = make_lu_dcmp(be, dnv)

    # Local matrix function
    array = be.local()

    def _pre_blusgs(i_begin, i_end, fnorm_vol, dt, diag, fjmat):
        # Compute digonal matrix
        for idx in range(i_begin, i_end):
            # Temporal diagonal matrix
            dmat = array((dnv, dnv), np.float64)

            # Initialize
            for row in range(dnv):
                for col in range(dnv):
                    dmat[row][col] = 0.0

            # Computes diagonal matrix based on neighbor cells
            for jdx in range(nface):
                fv = fnorm_vol[jdx, idx]
                for row in range(dnv):
                    for col in range(dnv):
                        dmat[row][col] += fjmat[0, row, col, jdx, idx]*fv
            
            # Complete implicit operator
            for kdx in range(dnv):
                dmat[kdx][kdx] += 1/(dt[idx]*factor)
            
            # LU decomposition for inverse process
            dcmp_func(dmat)

            # Allocate temporal matrix to diag array
            for row in range(dnv):
                for col in range(dnv):
                    diag[row, col, idx] = dmat[row][col]

    return _pre_blusgs


def make_tpre_blusgs(be, ele, nv, dsrcf, factor):
    # Number of faces
    nface = ele.nface

    # Number of variables
    dnv = int(nv[1] - nv[0])

    # Local matrix function
    array = be.local()

    # LU decomposition function
    dcmp_func = make_lu_dcmp(be, dnv)

    def _pre_tblusgs(i_begin, i_end, fnorm_vol, uptsb, dt, tdiag, tfjmat, dsrc):
        # Compute digonal matrix
        for idx in range(i_begin, i_end):
            # Allocate temporal turbulent diagonal matrix
            tmat = array((dnv, dnv), np.float64)

            # Initialize
            for row in range(dnv):
                for col in range(dnv):
                    tmat[row][col] = 0.0
            
            # Get conservative variables if needed
            u = uptsb[:, idx]

            # Computes diagonal matrix based on neighbor cells
            for jdx in range(nface):
                fv = fnorm_vol[jdx, idx]
                for row in range(dnv):
                    for col in range(dnv):
                        tmat[row][col] += tfjmat[0, row, col, jdx, idx]*fv
            
            # Compute Source term Jacobian
            dsrcf(u, tmat, dsrc[:, idx])
            
            # Complete implicit operator
            for kdx in range(dnv):
                tmat[kdx][kdx] += 1/(dt[idx]*factor)

            # LU decomposition for inverse process
            dcmp_func(tmat)

            # Allocate temporal matrix to digonal array
            for row in range(dnv):
                for col in range(dnv):
                    tdiag[row, col, idx] = tmat[row][col]

    return _pre_tblusgs


def make_serial_blusgs(be, ele, nv):
    # Local array and matrix
    array = be.local()

    # Get element attributes
    nface = ele.nface
    dnv = int(nv[1] - nv[0])
    nv0 = nv[0]

    # Forward/Backward substitution function
    sub_func = make_substitution(be, dnv)

    def _lower_sweep(i_begin, i_end, fnorm_vol, nei_ele, rhsb, dub, diag, fjmat):
        # Lower (Forward) sweep
        for idx in range(i_begin, i_end):
            rhs = array((dnv,), np.float64)
            dmat = array((dnv, dnv), np.float64)

            for row in range(dnv):
                for col in range(dnv):
                    dmat[row][col] = diag[row, col, idx]

            # Initialize rhs array with RHS
            for k in range(dnv):
                rhs[k] = rhsb[k+nv0, idx]

            for jdx in range(nface):
                neib = nei_ele[jdx, idx]

                if neib != idx:
                    fv = fnorm_vol[jdx, idx]
                    for kdx in range(dnv):
                        val = 0.0
                        for ldx in range(dnv):
                            val += fjmat[1, kdx, ldx, jdx, idx] * dub[ldx+nv0, neib]
                        rhs[kdx] -= val*fv
                        
            # Compute inverse of diagonal matrix multiplication
            sub_func(dmat, rhs)

            # Update dub array
            for kdx in range(dnv):
                dub[kdx+nv0, idx] = rhs[kdx]

    def _upper_sweep(i_begin, i_end, fnorm_vol, nei_ele, rhsb, dub, diag, fjmat):
        # Upper (Backward) sweep
        for idx in range(i_end-1, i_begin-1, -1):
            rhs = array((dnv,), np.float64)
            dmat = array((dnv, dnv), np.float64)

            for row in range(dnv):
                for col in range(dnv):
                    dmat[row][col] = diag[row, col, idx]

            # Initialize rhs array with RHS
            for k in range(dnv):
                rhs[k] = rhsb[k+nv0, idx]

            for jdx in range(nface):
                neib = nei_ele[jdx, idx]

                if neib != idx:
                    fv = fnorm_vol[jdx, idx]
                    for kdx in range(dnv):
                        val = 0.0
                        for ldx in range(dnv):
                            val += fjmat[1, kdx, ldx, jdx, idx] * dub[ldx+nv0, neib]
                        rhs[kdx] -= val*fv

            # Compute inverse of diagonal matrix multiplication
            sub_func(dmat, rhs)

            # Update dub array
            for kdx in range(dnv):
                dub[kdx+nv0, idx] = rhs[kdx]

    return _lower_sweep, _upper_sweep


def make_colored_blusgs(be, ele, nv):
    # Make local array
    array = be.local()

    # Get element attributes
    nface = ele.nface
    dnv = int(nv[1] - nv[0])
    nv0 = nv[0]

    # Matrix inverse - vector multiplication
    sub_func = make_substitution(be, dnv)

    def _sweep(i_begin, i_end, fnorm_vol, nei_ele, icolor, lcolor,
               rhsb, dub, diag, fjmat):
        for _idx in range(i_begin, i_end):
            # Lower sweep with coloring
            idx = icolor[_idx]
            curr_level = lcolor[idx]
            rhs = array((dnv,), np.float64)
            dmat = array((dnv, dnv), np.float64)

            for row in range(dnv):
                for col in range(dnv):
                    dmat[row][col] = diag[row, col, idx]

            # Initialize rhs array with RHS
            for k in range(dnv):
                rhs[k] = rhsb[k+nv0, idx]

            for jdx in range(nface):
                neib = nei_ele[jdx, idx]

                if lcolor[neib] != curr_level:
                    fv = fnorm_vol[jdx, idx]
                    for kdx in range(dnv):
                        val = 0.0
                        for ldx in range(dnv):
                            val += fjmat[1, kdx, ldx, jdx, idx] * dub[ldx+nv0, neib]
                        rhs[kdx] -= val*fv

            # Compute inverse of diagonal matrix multiplication
            sub_func(dmat, rhs)

            # Update dub array
            for kdx in range(dnv):
                dub[kdx+nv0, idx] = rhs[kdx]

    return _sweep


def make_diff_solution(res_idx):

    def _run(i_begin, i_end, dub, dubp1, subres):
        for idx in range(i_begin, i_end):
            tmp = dubp1[idx] - dub[res_idx, idx]
            subres[idx] = tmp * tmp
            dubp1[idx] = dub[res_idx, idx]
    
    return _run