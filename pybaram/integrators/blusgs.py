from pybaram.utils.inverse import make_lu_dcmp, make_substitution

import numpy as np


def make_rank_blusgs_pack(ele):
    nvars, nfvars = ele.nvars, ele.nfvars

    def _pack(i_begin, i_end, cell_ids, rhs, dt, rank_rhs, diag, a0):
        for idx in range(i_begin, i_end):
            ridx = cell_ids[idx]

            # Pack element-local RHS into rank-cell storage.
            for kdx in range(nvars):
                rank_rhs[kdx, ridx] = rhs[kdx, idx]

            # Start each flow diagonal block from the time contribution.
            for row in range(nfvars):
                for col in range(nfvars):
                    diag[row, col, ridx] = 0.0
                diag[row, row, ridx] = 1/dt[idx] + a0

    return _pack


def make_rank_blusgs_diag_pack(ele):
    nfvars = ele.nfvars

    def _pack(i_begin, i_end, cell_ids, dt, diag, a0):
        for idx in range(i_begin, i_end):
            ridx = cell_ids[idx]

            # Colored BLU-SGS packs RHS directly from element storage during
            # the sweep, so this kernel initializes only the diagonal block.
            for row in range(nfvars):
                for col in range(nfvars):
                    diag[row, col, ridx] = 0.0
                diag[row, row, ridx] = 1/dt[idx] + a0

    return _pack


def make_rank_tblusgs_pack(ele, dsrcf, factor=1.0):
    nturbvars = ele.nturbvars

    def _pack(i_begin, i_end, cell_ids, upts, dt, dsrc, diag, a0):
        for idx in range(i_begin, i_end):
            ridx = cell_ids[idx]

            # Source Jacobian contributes to the turbulence diagonal block.
            for row in range(nturbvars):
                for col in range(nturbvars):
                    diag[row, col, ridx] = 0.0

            dsrcf(upts[:, idx], diag[:, :, ridx], dsrc[:, idx])

            for row in range(nturbvars):
                diag[row, row, ridx] += 1/(dt[idx]*factor) + a0

    return _pack


def make_rank_blusgs_update(ele):
    nvars = ele.nvars

    def _update(i_begin, i_end, cell_ids, upts, du):
        for idx in range(i_begin, i_end):
            ridx = cell_ids[idx]

            # Scatter rank-cell corrections back to element-local storage.
            for kdx in range(nvars):
                upts[kdx, idx] += du[kdx, ridx]

    return _update


def make_rank_sub_residual(ele):
    nvars = ele.nvars

    def _subres(i_begin, i_end, cell_ids, vol, du, dup, res):
        for idx in range(i_begin, i_end):
            ridx = cell_ids[idx]

            # Store the correction change and keep a copy for the next
            # subiteration.
            for kdx in range(nvars):
                diff = du[kdx, ridx] - dup[kdx, ridx]
                res[kdx, idx] = diff*diff*vol[idx]
                dup[kdx, ridx] = du[kdx, ridx]

    return _subres


def make_rank_pre_blusgs(be, nvars):
    array = be.local()
    dcmp_func = make_lu_dcmp(be, nvars)

    def _pre(i_begin, i_end, face_indptr, face_slots, face_sides,
             face_area, rcp_vol, diag, jmat):
        for ridx in range(i_begin, i_end):
            dmat = array((nvars, nvars), np.float64)

            for row in range(nvars):
                for col in range(nvars):
                    dmat[row, col] = diag[row, col, ridx]

            # Add same-cell face Jacobian contributions to the diagonal block.
            for pos in range(face_indptr[ridx], face_indptr[ridx + 1]):
                slot = face_slots[pos]
                side = face_sides[pos]
                fv = face_area[slot]*rcp_vol[ridx]

                for row in range(nvars):
                    for col in range(nvars):
                        if side == 1:
                            val = jmat[0, row, col, slot]
                        else:
                            val = -jmat[1, row, col, slot]
                        dmat[row, col] += val*fv

            # Reuse diag storage for the LU factors.
            dcmp_func(dmat)

            for row in range(nvars):
                for col in range(nvars):
                    diag[row, col, ridx] = dmat[row, col]

    return _pre


def make_ele_pre_blusgs(be, ele, nvars):
    nface = ele.nface
    array = be.local()
    dcmp_func = make_lu_dcmp(be, nvars)

    def _pre(i_begin, i_end, cell_ids, face_refs, face_factors,
             diag, offdiag, jmat):
        for idx in range(i_begin, i_end):
            ridx = cell_ids[idx]
            dmat = array((nvars, nvars), np.float64)

            for row in range(nvars):
                for col in range(nvars):
                    dmat[row, col] = diag[row, col, ridx]

            for fidx in range(nface):
                # Signed refs map element-local faces to shared rank face
                # slots and encode face orientation.
                face = face_refs[fidx, idx]
                if face > 0:
                    slot = face - 1
                else:
                    slot = -face - 1
                fv = face_factors[fidx, idx]

                for row in range(nvars):
                    for col in range(nvars):
                        if face > 0:
                            dval = jmat[0, row, col, slot]
                            oval = jmat[1, row, col, slot]
                        else:
                            dval = -jmat[1, row, col, slot]
                            oval = -jmat[0, row, col, slot]

                        dmat[row, col] += dval*fv
                        offdiag[row, col, fidx, idx] = oval*fv

            # Reuse diag storage for the LU factors; offdiag is cached for
            # colored sweeps.
            dcmp_func(dmat)

            for row in range(nvars):
                for col in range(nvars):
                    diag[row, col, ridx] = dmat[row, col]

    return _pre


def make_rank_blusgs_sweep(be, var0, nvars):
    array = be.local()
    sub_func = make_substitution(be, nvars)

    def _lower(i_begin, i_end, face_indptr, face_slots, face_sides,
               face_neighbors, face_area, rcp_vol, rhsb, dub, diag,
               jmat):
        for ridx in range(i_begin, i_end):
            rhs = array((nvars,), np.float64)
            dmat = array((nvars, nvars), np.float64)

            for row in range(nvars):
                rhs[row] = rhsb[var0 + row, ridx]
                for col in range(nvars):
                    dmat[row, col] = diag[row, col, ridx]

            # Accumulate lower off-diagonal contributions.
            for pos in range(face_indptr[ridx], face_indptr[ridx + 1]):
                neib = face_neighbors[pos]
                slot = face_slots[pos]
                side = face_sides[pos]
                fv = face_area[slot]*rcp_vol[ridx]

                for row in range(nvars):
                    val = 0.0
                    for col in range(nvars):
                        if side == 1:
                            jval = jmat[1, row, col, slot]
                        else:
                            jval = -jmat[0, row, col, slot]
                        val += jval*dub[var0 + col, neib]
                    rhs[row] -= val*fv

            # Apply the lower-sweep block Gauss-Seidel correction.
            sub_func(dmat, rhs)
            for row in range(nvars):
                dub[var0 + row, ridx] = rhs[row]

    def _upper(i_begin, i_end, face_indptr, face_slots, face_sides,
               face_neighbors, face_area, rcp_vol, rhsb, dub, diag,
               jmat):
        for ridx in range(i_end - 1, i_begin - 1, -1):
            rhs = array((nvars,), np.float64)
            dmat = array((nvars, nvars), np.float64)

            for row in range(nvars):
                rhs[row] = rhsb[var0 + row, ridx]
                for col in range(nvars):
                    dmat[row, col] = diag[row, col, ridx]

            # Accumulate upper off-diagonal contributions.
            for pos in range(face_indptr[ridx], face_indptr[ridx + 1]):
                neib = face_neighbors[pos]
                slot = face_slots[pos]
                side = face_sides[pos]
                fv = face_area[slot]*rcp_vol[ridx]

                for row in range(nvars):
                    val = 0.0
                    for col in range(nvars):
                        if side == 1:
                            jval = jmat[1, row, col, slot]
                        else:
                            jval = -jmat[0, row, col, slot]
                        val += jval*dub[var0 + col, neib]
                    rhs[row] -= val*fv

            # Apply the upper-sweep block Gauss-Seidel correction.
            sub_func(dmat, rhs)
            for row in range(nvars):
                dub[var0 + row, ridx] = rhs[row]

    return _lower, _upper


def make_ele_colored_blusgs_sweep(be, ele, var0, nvars):
    nface = ele.nface
    array = be.local()
    sub_func = make_substitution(be, nvars)

    def _sweep(i_begin, i_end, color_order, cell_ids, face_neighbors,
               rhsb, dub, diag, offdiag):
        for pos in range(i_begin, i_end):
            # color_order slices contain element-local cells of one rank color.
            idx = color_order[pos]
            ridx = cell_ids[idx]
            rhs = array((nvars,), np.float64)
            dmat = array((nvars, nvars), np.float64)

            for row in range(nvars):
                rhs[row] = rhsb[var0 + row, idx]
                for col in range(nvars):
                    dmat[row, col] = diag[row, col, ridx]

            # Accumulate colored off-diagonal contributions cached in pre.
            for fidx in range(nface):
                neib = face_neighbors[fidx, idx]
                if neib >= 0:
                    for row in range(nvars):
                        val = 0.0
                        for col in range(nvars):
                            val += (
                                offdiag[row, col, fidx, idx]
                                * dub[var0 + col, neib]
                            )
                        rhs[row] -= val

            # Apply the colored block Gauss-Seidel correction.
            sub_func(dmat, rhs)
            for row in range(nvars):
                dub[var0 + row, ridx] = rhs[row]

    return _sweep
