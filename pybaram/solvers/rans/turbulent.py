# -*- coding: utf-8 -*-
import numpy as np


def make_vorticity(be, cplargs):
    # Compute vorticity

    ndims = cplargs['ndims']

    if ndims == 2:
        def vorticity(uc, gc):
            inv_rho = 1/uc[0]
            u = uc[1] * inv_rho
            v = uc[2] * inv_rho

            rho_x = gc[0][0]
            rho_y = gc[1][0]

            u_y = gc[1][1] - u*rho_y
            v_x = gc[0][2] - v*rho_x 

            # Compute mean rotation rate and its magnitude
            w_xy = (u_y - v_x)*inv_rho
            return abs(w_xy)

    else:
        def vorticity(uc, gc):
            inv_rho = 1/uc[0]
            u = uc[1] * inv_rho
            v = uc[2] * inv_rho
            w = uc[3] * inv_rho

            rho_x = gc[0][0]
            rho_y = gc[1][0]
            rho_z = gc[2][0]

            u_y = gc[1][1] - u*rho_y
            u_z = gc[2][1] - u*rho_z

            v_x = gc[0][2] - v*rho_x
            v_z = gc[2][2] - v*rho_z

            w_x = gc[0][3] - w*rho_x
            w_y = gc[1][3] - w*rho_y

            # Compute mean rotation rate and its magnitude
            w_xy = (u_y - v_x)*inv_rho
            w_yz = (v_z - w_y)*inv_rho
            w_zx = (w_x - u_z)*inv_rho
            return np.sqrt(w_xy**2 + w_yz**2 + w_zx**2)

    # Compile
    return be.compile(vorticity)


def make_strain_rate_mag(be, cplargs):
    ndims = cplargs['ndims']

    if ndims == 2:
        def strain_rate_mag(uc, gc):
            inv_rho = 1/uc[0]
            u = uc[1] * inv_rho
            v = uc[2] * inv_rho

            rho_x = gc[0][0]
            rho_y = gc[1][0]

            u_x = (gc[0][1] - u*rho_x)*inv_rho
            u_y = (gc[1][1] - u*rho_y)*inv_rho
            v_x = (gc[0][2] - v*rho_x)*inv_rho
            v_y = (gc[1][2] - v*rho_y)*inv_rho

            s_xy = 0.5*(u_y + v_x)
            return np.sqrt(2*(u_x**2 + v_y**2 + 2*s_xy**2))

    else:
        def strain_rate_mag(uc, gc):
            inv_rho = 1/uc[0]
            u = uc[1] * inv_rho
            v = uc[2] * inv_rho
            w = uc[3] * inv_rho

            rho_x = gc[0][0]
            rho_y = gc[1][0]
            rho_z = gc[2][0]

            u_x = (gc[0][1] - u*rho_x)*inv_rho
            u_y = (gc[1][1] - u*rho_y)*inv_rho
            u_z = (gc[2][1] - u*rho_z)*inv_rho

            v_x = (gc[0][2] - v*rho_x)*inv_rho
            v_y = (gc[1][2] - v*rho_y)*inv_rho
            v_z = (gc[2][2] - v*rho_z)*inv_rho

            w_x = (gc[0][3] - w*rho_x)*inv_rho
            w_y = (gc[1][3] - w*rho_y)*inv_rho
            w_z = (gc[2][3] - w*rho_z)*inv_rho

            s_xy = 0.5*(u_y + v_x)
            s_yz = 0.5*(v_z + w_y)
            s_zx = 0.5*(w_x + u_z)

            return np.sqrt(2*(u_x**2 + v_y**2 + w_z**2
                              + 2*(s_xy**2 + s_yz**2 + s_zx**2)))

    return be.compile(strain_rate_mag)
