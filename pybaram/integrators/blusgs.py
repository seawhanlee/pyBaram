from pybaram.utils.inverse import make_lu_dcmp, make_substitution

import numpy as np


def make_blusgs_update(ele):
    # Number of variables.
    nvars = ele.nvars

    def _update(i_begin, i_end, uptsb, dub):
        for idx in range(i_begin, i_end):
            for kdx in range(nvars):
                # Update solution by adding the correction.
                uptsb[kdx, idx] += dub[kdx, idx]

    return _update


def make_pre_blusgs(be, ele, nv, factor=1.0, a0=0.0):
    # Number of faces.
    nface = ele.nface

    # Number of variables.
    dnv = int(nv[1] - nv[0])

    # LU decomposition function.
    dcmp_func = make_lu_dcmp(be, dnv)

    # Local matrix factory.
    array = be.local()

    def _pre_blusgs(i_begin, i_end, fnorm_vol, dt, diag, fjmat):
        # Compute the diagonal block.
        for idx in range(i_begin, i_end):
            # Temporary diagonal block.
            dmat = array((dnv, dnv), np.float64)

            # Initialize the temporary block.
            for row in range(dnv):
                for col in range(dnv):
                    dmat[row][col] = 0.0

            # Accumulate diagonal contributions from cell faces.
            for jdx in range(nface):
                fv = fnorm_vol[jdx, idx]
                for row in range(dnv):
                    for col in range(dnv):
                        dmat[row][col] += fjmat[0, row, col, jdx, idx]*fv

            # Complete implicit operator
            for kdx in range(dnv):
                dmat[kdx][kdx] += 1/(dt[idx]*factor) + a0

            # LU decomposition for the substitution step.
            dcmp_func(dmat)

            # Store the LU-decomposed diagonal block.
            for row in range(dnv):
                for col in range(dnv):
                    diag[row, col, idx] = dmat[row][col]

    return _pre_blusgs


def make_tpre_blusgs(be, ele, nv, dsrcf, factor=1.0, a0=0.0):
    # Number of faces.
    nface = ele.nface

    # Number of variables.
    dnv = int(nv[1] - nv[0])

    # Local matrix factory.
    array = be.local()

    # LU decomposition function.
    dcmp_func = make_lu_dcmp(be, dnv)

    def _pre_tblusgs(i_begin, i_end, fnorm_vol, uptsb, dt, tdiag, tfjmat, dsrc):
        # Compute the turbulent diagonal block.
        for idx in range(i_begin, i_end):
            # Temporary turbulent diagonal block.
            tmat = array((dnv, dnv), np.float64)

            # Initialize the temporary block.
            for row in range(dnv):
                for col in range(dnv):
                    tmat[row][col] = 0.0

            # Get conservative variables for source Jacobian evaluation.
            u = uptsb[:, idx]

            # Accumulate diagonal contributions from cell faces.
            for jdx in range(nface):
                fv = fnorm_vol[jdx, idx]
                for row in range(dnv):
                    for col in range(dnv):
                        tmat[row][col] += tfjmat[0, row, col, jdx, idx]*fv

            # Add source-term Jacobian contribution.
            dsrcf(u, tmat, dsrc[:, idx])

            # Complete implicit operator
            for kdx in range(dnv):
                tmat[kdx][kdx] += 1/(dt[idx]*factor) + a0

            # LU decomposition for the substitution step.
            dcmp_func(tmat)

            # Store the LU-decomposed turbulent diagonal block.
            for row in range(dnv):
                for col in range(dnv):
                    tdiag[row, col, idx] = tmat[row][col]

    return _pre_tblusgs


def make_serial_blusgs(be, ele, nv):
    # Local array and matrix factory.
    array = be.local()

    # Element attributes.
    nface = ele.nface
    dnv = int(nv[1] - nv[0])
    nv0 = nv[0]

    # Forward/backward substitution function.
    sub_func = make_substitution(be, dnv)

    def _lower_sweep(i_begin, i_end, fnorm_vol, nei_ele, rhsb, dub, diag, fjmat):
        # Lower (forward) sweep.
        for idx in range(i_begin, i_end):
            rhs = array((dnv,), np.float64)
            dmat = array((dnv, dnv), np.float64)

            for row in range(dnv):
                for col in range(dnv):
                    dmat[row][col] = diag[row, col, idx]

            # Initialize the RHS work array.
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

            # Apply the inverse diagonal block.
            sub_func(dmat, rhs)

            # Update the correction array.
            for kdx in range(dnv):
                dub[kdx+nv0, idx] = rhs[kdx]

    def _upper_sweep(i_begin, i_end, fnorm_vol, nei_ele, rhsb, dub, diag, fjmat):
        # Upper (backward) sweep.
        for idx in range(i_end-1, i_begin-1, -1):
            rhs = array((dnv,), np.float64)
            dmat = array((dnv, dnv), np.float64)

            for row in range(dnv):
                for col in range(dnv):
                    dmat[row][col] = diag[row, col, idx]

            # Initialize the RHS work array.
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

            # Apply the inverse diagonal block.
            sub_func(dmat, rhs)

            # Update the correction array.
            for kdx in range(dnv):
                dub[kdx+nv0, idx] = rhs[kdx]

    return _lower_sweep, _upper_sweep


def make_colored_blusgs(be, ele, nv):
    # Local array and matrix factory.
    array = be.local()

    # Element attributes.
    nface = ele.nface
    dnv = int(nv[1] - nv[0])
    nv0 = nv[0]

    # Forward/backward substitution function.
    sub_func = make_substitution(be, dnv)

    def _sweep(i_begin, i_end, fnorm_vol, nei_ele, icolor, lcolor,
               rhsb, dub, diag, fjmat):
        for _idx in range(i_begin, i_end):
            # Sweep cells in the current color level.
            idx = icolor[_idx]
            curr_level = lcolor[idx]
            rhs = array((dnv,), np.float64)
            dmat = array((dnv, dnv), np.float64)

            for row in range(dnv):
                for col in range(dnv):
                    dmat[row][col] = diag[row, col, idx]

            # Initialize the RHS work array.
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

            # Apply the inverse diagonal block.
            sub_func(dmat, rhs)

            # Update the correction array.
            for kdx in range(dnv):
                dub[kdx+nv0, idx] = rhs[kdx]

    return _sweep


def make_sub_residual(ele):
    nvars = ele.nvars
    
    def _kern(i_begin, i_end, vol, du, dup, res):
        for idx in range(i_begin, i_end):
            for j in range(nvars):
                res[j, idx] = (du[j, idx] - dup[j, idx])**2*vol[idx]
                dup[j, idx] = du[j, idx]

    return _kern
