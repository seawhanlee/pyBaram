from pybaram.utils.inverse import make_lu_dcmp, make_substitution
from pybaram.utils.nb import dot
import numpy as np


def make_jacobi_update(ele):
    # Number of variables
    nvars = ele.nvars

    # Update next time step solution
    def _update(i_begin, i_end, uptsb, dub, subres):
        for idx in range(i_begin, i_end):
            for kdx in range(nvars):
                # Update solution by adding residual
                uptsb[kdx, idx] += dub[kdx, idx]

                # Initialize dub array
                dub[kdx, idx] = 0.0
            
            # Initialize subres array
            subres[idx] = 0.0

    return _update


def make_pre_jacobi(be, ele, nv, factor=1.0):
    # Number of faces
    nface = ele.nface

    # Number of variables
    dnv = nv[1] - nv[0]

    # Normal vectors at faces
    fnorm_vol = ele.mag_fnorm * ele.rcp_vol

    # LU decomposition function
    dcmp_func = make_lu_dcmp(be, dnv)

    # Local matrix function
    matrix = be.local_matrix()

    def _pre_diag(i_begin, i_end, dt, diag, fjmat):
        # Compute diagonal matrix
        for idx in range(i_begin, i_end):
            # Temporal diagonal matrix
            dmat = matrix(dnv*dnv, (dnv, dnv))

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

    return _pre_diag


def make_tpre_jacobi(be, ele, nv, dsrcf, factor):
    # Number of faces
    nface = ele.nface

    # Number of variables
    dnv = nv[1] - nv[0]

    # Normal vectors at faces
    fnorm_vol = ele.mag_fnorm * ele.rcp_vol

    # Local matrix function
    matrix = be.local_matrix()
    
    def _pre_tdiag(i_begin, i_end, uptsb, dt, tdiag, tfjmat, dsrc):
        # Compute digonal matrix
        for idx in range(i_begin, i_end):
            # Allocate temporal turbulent diagonal matrix
            tmat = matrix(dnv*dnv, (dnv, dnv))

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

            # Allocate temporal matrix to digonal array
            for row in range(dnv):
                for col in range(dnv):
                    tdiag[row, col, idx] = tmat[row][col]

    return _pre_tdiag


def make_jacobi_sweep(be, ele, nv):
    # Local array and matrix
    array = be.local_array()
    matrix = be.local_matrix()

    # Get element attributes
    nface = ele.nface
    dnv = nv[1] - nv[0]

    # Get index array for neihboring cells
    nei_ele = ele.nei_ele

    # Normal vectors at faces
    fnorm_vol= ele.mag_fnorm * ele.rcp_vol

    # Forward/Backward substitution function
    sub_func = make_substitution(be, dnv)

    def _jacobi_sweep(i_begin, i_end, rhsb, dub, rod, fjmat):
        # Compute R-(L+U)x
        for idx in range(i_begin, i_end):
            rhs = array(dnv)

            # Initialize rhs array with RHS
            for kdx in range(dnv):
                rhs[kdx] = rhsb[kdx+nv[0], idx]

            # Computes Jacobian matrix based on neighbor cells
            for jdx in range(nface):
                neib = nei_ele[jdx, idx]

                if neib != idx:
                    fv = fnorm_vol[jdx, idx]
                    neimat = fjmat[1, :, :, jdx, idx]

                    for kdx in range(dnv):
                        rhs[kdx] -= dot(neimat[kdx, :], dub[:, neib], dnv, 0, nv[0]) * fv

            # Allocates to each rod array
            for kdx in range(dnv):
                rod[kdx+nv[0], idx] = rhs[kdx]
        
    def _jacobi_compute(i_begin, i_end, dub, rod, diag):
        # Compute Ax = b
        for idx in range(i_begin, i_end):
            rhs = array(dnv)
            dmat = matrix(dnv*dnv, (dnv, dnv))

            for row in range(dnv):
                for col in range(dnv):
                    dmat[row][col] = diag[row, col, idx]

            # Reallocate rod element value to rhs array
            for kdx in range(dnv):
                rhs[kdx] = rod[kdx+nv[0], idx]
            
            sub_func(dmat, rhs)
            
            # Inner-update dub array
            for kdx in range(dnv):
                dub[kdx+nv[0], idx] = rhs[kdx]

    return _jacobi_sweep, _jacobi_compute


