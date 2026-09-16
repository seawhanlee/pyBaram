import numpy as np


def make_lusgs_pack(ele, turb_factor=1.0):
    nvars, nfvars = ele.nvars, ele.nfvars

    def _pack(i_begin, i_end, cell_ids, upts, rhs, dt, dsrc,
              rank_u, rank_du, rank_diag, a0):
        for idx in range(i_begin, i_end):
            ridx = cell_ids[idx]

            for kdx in range(nvars):
                factor = 1.0 if kdx < nfvars else turb_factor

                # Convert element-local SOA data into rank-cell storage.
                rank_u[kdx, ridx] = upts[kdx, idx]
                rank_du[kdx, ridx] = rhs[kdx, idx]
                rank_diag[kdx, ridx] = (
                    1/(dt[idx]*factor) + a0 + dsrc[kdx, idx]
                )

    return _pack


def make_rank_lusgs_update(ele):
    nvars = ele.nvars

    def _update(i_begin, i_end, cell_ids, upts, rank_du):
        for idx in range(i_begin, i_end):
            ridx = cell_ids[idx]

            for kdx in range(nvars):
                # Scatter the rank-cell correction back to the element.
                upts[kdx, idx] += rank_du[kdx, ridx]

    return _update


def make_diff_flux(nvars, dnv, fluxf, array):
    # Difference of flux vectors.
    def _diff_flux(u, du, df, nf):
        f = array((dnv,), np.float64)
        for i in range(nvars):
            du[i] += u[i]

        fluxf(u, nf, f)
        fluxf(du, nf, df)

        for i in range(dnv):
            df[i] -= f[i]

    return _diff_flux


def make_rank_lusgs_common(nv, kappa=1.01):
    nv0, nv1 = nv

    def _pre_lusgs(i_begin, i_end, face_indptr, face_slots, face_area,
                   rcp_vol, diag, lambdaf):
        for ridx in range(i_begin, i_end):
            spectral_diag = 0.0

            # CSR entries point from a rank cell to its physical face slots.
            for pos in range(face_indptr[ridx], face_indptr[ridx + 1]):
                slot = face_slots[pos]
                lamf = lambdaf[slot]*kappa
                spectral_diag += (
                    0.5*lamf*face_area[slot]*rcp_vol[ridx]
                )

            for kdx in range(nv0, nv1):
                diag[kdx, ridx] += spectral_diag

    return _pre_lusgs


def make_ele_lusgs_common(ele, nv, kappa=1.01):
    nface = ele.nface
    nv0, nv1 = nv

    def _pre_lusgs(i_begin, i_end, cell_ids, face_refs, face_factors,
                   diag, wave_factors, lambdaf):
        for idx in range(i_begin, i_end):
            ridx = cell_ids[idx]
            spectral_diag = 0.0

            for fidx in range(nface):
                # Signed refs map element-local faces to shared rank face
                # slots; the sign stores face orientation for other kernels.
                face = face_refs[fidx, idx]
                if face > 0:
                    slot = face - 1
                else:
                    slot = -face - 1

                # Cache kappa*lambda*A/V for colored sweeps.
                wave = kappa*lambdaf[slot]*face_factors[fidx, idx]
                wave_factors[fidx, idx] = wave
                spectral_diag += 0.5*wave

            for kdx in range(nv0, nv1):
                diag[kdx, ridx] += spectral_diag

    return _pre_lusgs


def make_rank_lusgs_sweep(be, nvars, nv, flux, kappa=1.01):
    dnv = int(nv[1] - nv[0])
    nv0, nv1 = nv
    array = be.local()
    diff_flux = be.compile(make_diff_flux(nvars, dnv, flux, array))

    def _lower(i_begin, i_end, face_indptr, face_slots, face_sides,
               face_neighbors, face_area, face_normal, rcp_vol, rank_u,
               rank_du, diag, lambdaf):
        for ridx in range(i_begin, i_end):
            du = array((nvars,), np.float64)
            dfj = array((dnv,), np.float64)
            df = array((dnv,), np.float64)

            for kdx in range(dnv):
                df[kdx] = 0.0

            for pos in range(face_indptr[ridx], face_indptr[ridx + 1]):
                neib = face_neighbors[pos]
                if neib < ridx:
                    # Accumulate lower off-diagonal contributions.
                    slot = face_slots[pos]
                    u = rank_u[:, neib]
                    side = face_sides[pos]
                    nf = side*face_normal[:, slot]
                    fv = face_area[slot]*rcp_vol[ridx]

                    for kdx in range(nvars):
                        du[kdx] = 0.0
                    for kdx in range(nv0, nv1):
                        du[kdx] = rank_du[kdx, neib]

                    diff_flux(u, du, dfj, nf)

                    for kdx in range(dnv):
                        df[kdx] += (
                            dfj[kdx]
                            - kappa*lambdaf[slot]
                            * rank_du[kdx + nv0, neib]
                        )*fv

            for kdx in range(dnv):
                # Apply the lower-sweep Gauss-Seidel correction.
                rank_du[kdx + nv0, ridx] = (
                    rank_du[kdx + nv0, ridx] - 0.5*df[kdx]
                )/diag[kdx + nv0, ridx]

    def _upper(i_begin, i_end, face_indptr, face_slots, face_sides,
               face_neighbors, face_area, face_normal, rcp_vol, rank_u,
               rank_du, diag, lambdaf):
        for ridx in range(i_end - 1, i_begin - 1, -1):
            du = array((nvars,), np.float64)
            dfj = array((dnv,), np.float64)
            df = array((dnv,), np.float64)

            for kdx in range(dnv):
                df[kdx] = 0.0

            for pos in range(face_indptr[ridx], face_indptr[ridx + 1]):
                neib = face_neighbors[pos]
                if neib > ridx:
                    # Accumulate upper off-diagonal contributions.
                    slot = face_slots[pos]
                    u = rank_u[:, neib]
                    side = face_sides[pos]
                    nf = side*face_normal[:, slot]
                    fv = face_area[slot]*rcp_vol[ridx]

                    for kdx in range(nvars):
                        du[kdx] = 0.0
                    for kdx in range(nv0, nv1):
                        du[kdx] = rank_du[kdx, neib]

                    diff_flux(u, du, dfj, nf)

                    for kdx in range(dnv):
                        df[kdx] += (
                            dfj[kdx]
                            - kappa*lambdaf[slot]
                            * rank_du[kdx + nv0, neib]
                        )*fv

            for kdx in range(dnv):
                # Apply the upper-sweep Gauss-Seidel correction.
                rank_du[kdx + nv0, ridx] -= (
                    0.5*df[kdx]
                    / diag[kdx + nv0, ridx]
                )

    return _lower, _upper


def make_ele_colored_lusgs_sweep(be, ele, nvars, nv, flux):
    nface = ele.nface
    dnv = int(nv[1] - nv[0])
    nv0, nv1 = nv
    array = be.local()
    diff_flux = be.compile(make_diff_flux(nvars, dnv, flux, array))

    def _lower(i_begin, i_end, color_order, cell_ids, face_neighbors,
               face_factors, face_normals, wave_factors, rank_u,
               rank_du, diag):
        for pos in range(i_begin, i_end):
            # color_order slices contain element-local cells of one rank color.
            idx = color_order[pos]
            ridx = cell_ids[idx]
            df = array((dnv,), np.float64)

            for kdx in range(dnv):
                df[kdx] = 0.0

            for fidx in range(nface):
                neib = face_neighbors[fidx, idx]
                if neib >= 0:
                    # Accumulate lower off-diagonal contributions.
                    du = array((nvars,), np.float64)
                    dfj = array((dnv,), np.float64)

                    for kdx in range(nvars):
                        du[kdx] = 0.0
                    for kdx in range(nv0, nv1):
                        du[kdx] = rank_du[kdx, neib]

                    diff_flux(
                        rank_u[:, neib], du, dfj,
                        face_normals[:, fidx, idx]
                    )

                    # face_factors = A/V and wave_factors = kappa*lambda*A/V.
                    fv = face_factors[fidx, idx]
                    wave = wave_factors[fidx, idx]
                    for kdx in range(dnv):
                        df[kdx] += (
                            dfj[kdx]*fv
                            - wave*rank_du[kdx + nv0, neib]
                        )

            for kdx in range(dnv):
                # Apply the lower-sweep Gauss-Seidel correction.
                rank_du[kdx + nv0, ridx] = (
                    rank_du[kdx + nv0, ridx] - 0.5*df[kdx]
                )/diag[kdx + nv0, ridx]

    def _upper(i_begin, i_end, color_order, cell_ids, face_neighbors,
               face_factors, face_normals, wave_factors, rank_u,
               rank_du, diag):
        for pos in range(i_begin, i_end):
            # color_order slices contain element-local cells of one rank color.
            idx = color_order[pos]
            ridx = cell_ids[idx]
            df = array((dnv,), np.float64)

            for kdx in range(dnv):
                df[kdx] = 0.0

            for fidx in range(nface):
                neib = face_neighbors[fidx, idx]
                if neib >= 0:
                    # Accumulate upper off-diagonal contributions.
                    du = array((nvars,), np.float64)
                    dfj = array((dnv,), np.float64)

                    for kdx in range(nvars):
                        du[kdx] = 0.0
                    for kdx in range(nv0, nv1):
                        du[kdx] = rank_du[kdx, neib]

                    diff_flux(
                        rank_u[:, neib], du, dfj,
                        face_normals[:, fidx, idx]
                    )

                    # face_factors = A/V and wave_factors = kappa*lambda*A/V.
                    fv = face_factors[fidx, idx]
                    wave = wave_factors[fidx, idx]
                    for kdx in range(dnv):
                        df[kdx] += (
                            dfj[kdx]*fv
                            - wave*rank_du[kdx + nv0, neib]
                        )

            for kdx in range(dnv):
                # Apply the upper-sweep Gauss-Seidel correction.
                rank_du[kdx + nv0, ridx] -= (
                    0.5*df[kdx]/diag[kdx + nv0, ridx]
                )

    return _lower, _upper
