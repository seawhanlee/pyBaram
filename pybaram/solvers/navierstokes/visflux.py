# -*- coding: utf-8 -*-
def make_visflux(be, cplargs):
    ndims = cplargs['ndims']
    gamma, pr = cplargs['gamma'], cplargs['pr']
    is_axi = cplargs.get('axisymmetric', False)
    ridx = cplargs.get('axisymmetric_radius_idx', 1)

    def visflux2d(uf, gf, nf, mu, fn):
        inv_rho = 1/uf[0]
        u = uf[1]*inv_rho
        v = uf[2]*inv_rho
        e = uf[3]

        rho_x = gf[0][0]
        rho_y = gf[1][0]

        # rho Velocity derivatives
        u_x = gf[0][1] - u*rho_x
        u_y = gf[1][1] - u*rho_y
        v_x = gf[0][2] - v*rho_x
        v_y = gf[1][2] - v*rho_y

        e_x = gf[0][3]
        e_y = gf[1][3]

        # Temperature derivative (c_v*dt/d[x.y])
        t_x = inv_rho*(e_x - (inv_rho*rho_x*e + u*u_x + v*v_x))
        t_y = inv_rho*(e_y - (inv_rho*rho_y*e + u*u_y + v*v_y))

        # Stress tensor
        t_xx = 2*mu*inv_rho*(u_x - 1/3*(u_x + v_y))
        t_yy = 2*mu*inv_rho*(v_y - 1/3*(u_x + v_y))
        t_xy = mu*inv_rho*(v_x + u_y)

        fn[1] -= nf[0]*t_xx + nf[1]*t_xy
        fn[2] -= nf[0]*t_xy + nf[1]*t_yy
        fn[3] -= nf[0]*(u*t_xx + v*t_xy + gamma*(mu/pr)*t_x) + \
            nf[1]*(u*t_xy + v*t_yy + gamma*(mu/pr)*t_y)

    def visflux2d_axi(uf, gf, nf, mu, r, fn):
        inv_rho = 1/uf[0]
        u = uf[1]*inv_rho
        v = uf[2]*inv_rho
        e = uf[3]

        rho_x = gf[0][0]
        rho_y = gf[1][0]

        # rho Velocity derivatives
        u_x = gf[0][1] - u*rho_x
        u_y = gf[1][1] - u*rho_y
        v_x = gf[0][2] - v*rho_x
        v_y = gf[1][2] - v*rho_y

        e_x = gf[0][3]
        e_y = gf[1][3]

        # Temperature derivative (c_v*dt/d[x.y])
        t_x = inv_rho*(e_x - (inv_rho*rho_x*e + u*u_x + v*v_x))
        t_y = inv_rho*(e_y - (inv_rho*rho_y*e + u*u_y + v*v_y))

        vr = u if ridx == 0 else v

        # Stress tensor with axisymmetric divergence.
        divv = inv_rho*(u_x + v_y) + vr/r
        t_xx = 2*mu*(inv_rho*u_x - 1/3*divv)
        t_yy = 2*mu*(inv_rho*v_y - 1/3*divv)
        t_xy = mu*inv_rho*(v_x + u_y)

        fn[1] -= nf[0]*t_xx + nf[1]*t_xy
        fn[2] -= nf[0]*t_xy + nf[1]*t_yy
        fn[3] -= nf[0]*(u*t_xx + v*t_xy + gamma*(mu/pr)*t_x) + \
            nf[1]*(u*t_xy + v*t_yy + gamma*(mu/pr)*t_y)

    def visflux3d(uf, gf, nf, mu, fn):
        inv_rho = 1/uf[0]
        u = uf[1]*inv_rho
        v = uf[2]*inv_rho
        w = uf[3]*inv_rho
        e = uf[4]

        rho_x = gf[0][0]
        rho_y = gf[1][0]
        rho_z = gf[2][0]

        # rho Velocity derivatives
        u_x = gf[0][1] - u*rho_x
        u_y = gf[1][1] - u*rho_y
        u_z = gf[2][1] - u*rho_z

        v_x = gf[0][2] - v*rho_x
        v_y = gf[1][2] - v*rho_y
        v_z = gf[2][2] - v*rho_z

        w_x = gf[0][3] - w*rho_x
        w_y = gf[1][3] - w*rho_y
        w_z = gf[2][3] - w*rho_z

        e_x = gf[0][4]
        e_y = gf[1][4]
        e_z = gf[2][4]

        # Temperature derivative (c_v*dt/d[x.y])
        t_x = inv_rho*(e_x - (inv_rho*rho_x*e + u*u_x + v*v_x + w*w_x))
        t_y = inv_rho*(e_y - (inv_rho*rho_y*e + u*u_y + v*v_y + w*w_y))
        t_z = inv_rho*(e_z - (inv_rho*rho_z*e + u*u_z + v*v_z + w*w_z))

        # Stress tensor
        t_xx = 2*mu*inv_rho*(u_x - 1/3*(u_x + v_y + w_z))
        t_yy = 2*mu*inv_rho*(v_y - 1/3*(u_x + v_y + w_z))
        t_zz = 2*mu*inv_rho*(w_z - 1/3*(u_x + v_y + w_z))

        t_xy = mu*inv_rho*(v_x + u_y)
        t_yz = mu*inv_rho*(w_y + v_z)
        t_zx = mu*inv_rho*(u_z + w_x)

        fn[1] -= nf[0]*t_xx + nf[1]*t_xy + nf[2]*t_zx
        fn[2] -= nf[0]*t_xy + nf[1]*t_yy + nf[2]*t_yz
        fn[3] -= nf[0]*t_zx + nf[1]*t_yz + nf[2]*t_zz
        fn[4] -= nf[0]*(u*t_xx + v*t_xy + w*t_zx + gamma*(mu/pr)*t_x) + \
            nf[1]*(u*t_xy + v*t_yy + w*t_yz + gamma*(mu/pr)*t_y) +\
            nf[2]*(u*t_zx + v*t_yz + w*t_zz + gamma*(mu/pr)*t_z)

    if ndims == 2 and is_axi:
        return be.compile(visflux2d_axi)
    elif ndims == 2:
        return be.compile(visflux2d)
    elif ndims == 3:
        return be.compile(visflux3d)
