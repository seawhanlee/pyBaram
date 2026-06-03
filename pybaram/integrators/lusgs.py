import numpy as np


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


def make_lusgs_update(ele):
    # Number of variables.
    nvars = ele.nvars

    def _update(i_begin, i_end, uptsb, rhsb):
        # Update solution by adding the correction.
        for idx in range(i_begin, i_end):
            for kdx in range(nvars):
                uptsb[kdx, idx] += rhsb[kdx, idx]

    return _update


def make_lusgs_common(ele, a0=0.0, factor=1.0, kappa=1.01):
    # Number of faces.
    nface = ele.nface

    def _pre_lusgs(i_begin, i_end, fnorm_vol, dt, diag, lambdaf):
        # Construct diagonal terms for LU-SGS.
        for idx in range(i_begin, i_end):
            # Diagonal of the implicit operator.
            diag[idx] = 1 / (dt[idx]*factor) + a0

            for jdx in range(nface):
                # Diffusive margin of wave speed at a face.
                lamf = lambdaf[jdx, idx]*kappa

                # Save the inflated spectral radius.
                lambdaf[jdx, idx] = lamf

                # Add the lower/upper spectral-radius contribution.
                diag[idx] += 0.5*lamf*fnorm_vol[jdx, idx]

    return _pre_lusgs


def make_serial_lusgs(be, ele, nv, _flux):
    # Dimensions for variables and faces.
    nvars, nface = ele.nvars, ele.nface
    dnv = int(nv[1] - nv[0])
    nv0, nv1 = nv[0], nv[1]

    # Local array factory.
    array = be.local()

    # Pre-compile the flux-difference function.
    _diff_flux = be.compile(make_diff_flux(nvars, dnv, _flux, array))

    def _lower_sweep(i_begin, i_end, fnorm_vol, vec_fnorm, nei_ele,
                     uptsb, dub, diag, dsrc, lambdaf):
        # Lower sweep via mapping
        for idx in range(i_begin, i_end):
            du = array((nvars,), np.float64)
            dfj = array((dnv,), np.float64)
            df = array((dnv,), np.float64)

            for kdx in range(dnv):
                df[kdx] = 0.0

            for jdx in range(nface):
                # Compute the lower portion of the off-diagonal term.
                nf = vec_fnorm[jdx, :, idx]

                neib = nei_ele[jdx, idx]
                if neib < idx:
                    u = uptsb[:, neib]

                    for kdx in range(nvars):
                        du[kdx] = 0.0

                    for kdx in range(nv0, nv1):
                        du[kdx] = dub[kdx, neib]

                    _diff_flux(u, du, dfj, nf)

                    for kdx in range(dnv):
                        df[kdx] += (dfj[kdx] - lambdaf[jdx, idx]
                                    * dub[kdx+nv0, neib])*fnorm_vol[jdx, idx]

            for kdx in range(dnv):
                # Gauss-Seidel update with the lower portion.
                dub[kdx+nv0, idx] = (dub[kdx+nv0, idx] -
                                       0.5*df[kdx])/(diag[idx] + dsrc[kdx+nv0, idx])

    def _upper_sweep(i_begin, i_end, fnorm_vol, vec_fnorm, nei_ele,
                     uptsb, dub, diag, dsrc, lambdaf):
        for idx in range(i_end-1, i_begin-1, -1):
            # Upper sweep via mapping (reverse order)
            du = array((nvars,), np.float64)
            dfj = array((dnv,), np.float64)
            df = array((dnv,), np.float64)

            for kdx in range(dnv):
                df[kdx] = 0.0

            for jdx in range(nface):
                nf = vec_fnorm[jdx, :, idx]

                neib = nei_ele[jdx, idx]
                if neib > idx:
                    # Compute the upper portion of the off-diagonal term.
                    u = uptsb[:, neib]

                    for kdx in range(nvars):
                        du[kdx] = 0.0

                    for kdx in range(nv0, nv1):
                        du[kdx] = dub[kdx, neib]

                    _diff_flux(u, du, dfj, nf)

                    for kdx in range(dnv):
                        df[kdx] += (dfj[kdx] - lambdaf[jdx, idx]
                                    * dub[kdx+nv0, neib])*fnorm_vol[jdx, idx]

            for kdx in range(dnv):
                # Gauss-Seidel update with the upper portion.
                dub[kdx+nv0, idx] = dub[kdx+nv0, idx] - \
                    0.5*df[kdx]/(diag[idx] + dsrc[kdx+nv0, idx])

    return _lower_sweep, _upper_sweep


def make_colored_lusgs(be, ele, nv, _flux):
    # Dimensions for variables and faces.
    nvars, nface = ele.nvars, ele.nface
    dnv = int(nv[1] - nv[0])
    nv0, nv1 = nv[0], nv[1]

    # Local array factory.
    array = be.local()

    # Pre-compile the flux-difference function.
    _diff_flux = be.compile(make_diff_flux(nvars, dnv, _flux, array))

    def _lower_sweep(i_begin, i_end, fnorm_vol, vec_fnorm, nei_ele, icolor, lcolor,
                     uptsb, dub, diag, dsrc, lambdaf):
        for _idx in range(i_begin, i_end):
            # Lower sweep with coloring
            idx = icolor[_idx]
            curr_level = lcolor[idx]

            du = array((nvars,), np.float64)
            dfj = array((dnv,), np.float64)
            df = array((dnv,), np.float64)

            for kdx in range(dnv):
                df[kdx] = 0.0

            for jdx in range(nface):
                # Compute the lower portion of the off-diagonal term.
                nf = vec_fnorm[jdx, :, idx]

                neib = nei_ele[jdx, idx]
                if lcolor[neib] < curr_level:
                    u = uptsb[:, neib]

                    for kdx in range(nvars):
                        du[kdx] = 0.0

                    for kdx in range(nv0, nv1):
                        du[kdx] = dub[kdx, neib]

                    _diff_flux(u, du, dfj, nf)

                    for kdx in range(dnv):
                        df[kdx] += (dfj[kdx] - lambdaf[jdx, idx]
                                    * dub[kdx+nv0, neib])*fnorm_vol[jdx, idx]

            for kdx in range(dnv):
                # Gauss-Seidel update with the lower portion.
                dub[kdx+nv0, idx] = (dub[kdx+nv0, idx] -
                                       0.5*df[kdx])/(diag[idx] + dsrc[kdx+nv0, idx])

    def _upper_sweep(i_begin, i_end, fnorm_vol, vec_fnorm, nei_ele, icolor, lcolor,
                     uptsb, dub, diag, dsrc, lambdaf):
        for _idx in range(i_begin, i_end):
            # Upper sweep via coloring over reversed color levels.
            idx = icolor[_idx]
            curr_level = lcolor[idx]

            du = array((nvars,), np.float64)
            dfj = array((dnv,), np.float64)
            df = array((dnv,), np.float64)

            for kdx in range(dnv):
                df[kdx] = 0.0

            for jdx in range(nface):
                # Compute the upper portion of the off-diagonal term.
                nf = vec_fnorm[jdx, :, idx]

                neib = nei_ele[jdx, idx]
                if lcolor[neib] > curr_level:
                    u = uptsb[:, neib]

                    for kdx in range(nvars):
                        du[kdx] = 0.0

                    for kdx in range(nv0, nv1):
                        du[kdx] = dub[kdx, neib]

                    _diff_flux(u, du, dfj, nf)

                    for kdx in range(dnv):
                        df[kdx] += (dfj[kdx] - lambdaf[jdx, idx]
                                    * dub[kdx+nv0, neib])*fnorm_vol[jdx, idx]

            for kdx in range(dnv):
                # Gauss-Seidel update with the upper portion.
                dub[kdx+nv0, idx] = dub[kdx+nv0, idx] - \
                    0.5*df[kdx]/(diag[idx] + dsrc[kdx+nv0, idx])

    return _lower_sweep, _upper_sweep
