def get_viscous_jacobian(name, be, cplargs, sign='positive'):
    return eval('make_'+name+'_jacobian')(be, cplargs, sign)


def make_tlns_jacobian(be, cplargs, sign):
    """
    Thin Layer Navier-Stokes Jacobian
    Ref : 2005, Blazek, J., 
        "Computational Fluid Dynamics: Principles and Applications (2nd ed.)",
        Elsevier.
    """

    # Constants
    pr, gamma = cplargs['pr'], cplargs['gamma']
    ndims = cplargs['ndims']
    if sign == 'positive':
        op = 1.0
    elif sign == 'negative':
        op = -1.0
    else:
        raise ValueError("Wrong sign of viscous jacobian")
    
    def tlns2d(uf, nf, A, mu, rcp_dx):
        # Basic variables
        nx = nf[0]
        ny = nf[1]

        inv_rho = 1/uf[0]
        u = uf[1]*inv_rho
        v = uf[2]*inv_rho
        e = uf[3]*inv_rho
        v2 = u**2 + v**2

        # Metric terms
        a1 = 1.0 + nx**2/3.0
        a2 = nx*ny/3.0
        a3 = 1.0 + ny**2/3.0
        a4 = gamma/pr

        # Jacobian elements
        b21 = -(a1*u + a2*v)
        b31 = -(a2*u + a3*v)
        b41 = a4*(v2-e) - a1*u*u - 2*a2*u*v - a3*v*v
        b42 = -a4*u - b21
        b43 = -a4*v - b31

        # Computation
        mu *= op*rcp_dx*inv_rho
        A[1, 0] += mu*b21
        A[1, 1] += mu*a1
        A[1, 2] += mu*a2
        A[2, 0] += mu*b31
        A[2, 1] += mu*a2
        A[2, 2] += mu*a3
        A[3, 0] += mu*b41
        A[3, 1] += mu*b42
        A[3, 2] += mu*b43
        A[3, 3] += mu*a4

    def tlns3d(uf, nf, A, mu, rcp_dx):
        # Basic variables
        nx = nf[0]
        ny = nf[1]
        nz = nf[2]

        inv_rho = 1/uf[0]
        u = uf[1]*inv_rho
        v = uf[2]*inv_rho
        w = uf[3]*inv_rho
        e = uf[4]*inv_rho
        v2 = u**2 + v**2 + w**2

        # Jacobian elements
        a1 = 1.0 + nx**2/3.0
        a2 = nx*ny/3.0
        a3 = nx*nz/3.0
        a4 = 1.0 + ny**2/3.0
        a5 = ny*nz/3.0
        a6 = 1.0 + nz**2/3.0
        a7 = gamma/pr

        b21 = -(a1*u + a2*v + a3*w)
        b31 = -(a2*u + a4*v + a5*w)
        b41 = -(a3*u + a5*v + a6*w)
        b51 = a7*(v2-e) - a1*u*u - a4*v*v - a6*w*w \
                        - 2.0*(a2*u*v + a3*u*w + a5*v*w)
        b52 = -a7*u - b21
        b53 = -a7*v - b31
        b54 = -a7*w - b41

        # Computation
        mu *= op*rcp_dx*inv_rho
        A[1, 0] += mu*b21
        A[1, 1] += mu*a1
        A[1, 2] += mu*a2
        A[1, 3] += mu*a3
        A[2, 0] += mu*b31
        A[2, 1] += mu*a2
        A[2, 2] += mu*a4
        A[2, 3] += mu*a5
        A[3, 0] += mu*b41
        A[3, 1] += mu*a3
        A[3, 2] += mu*a5
        A[3, 3] += mu*a6
        A[4, 0] += mu*b51
        A[4, 1] += mu*b52
        A[4, 2] += mu*b53
        A[4, 3] += mu*b54
        A[4, 4] += mu*a7

    if ndims == 2:
        return be.compile(tlns2d)
    elif ndims == 3:
        return be.compile(tlns3d)


def make_approximate_jacobian(be, cplargs, sign):
    """
    Spectral radius on diagonal element
    """

    # Constants
    pr, gamma = cplargs['pr'], cplargs['gamma']
    nfvars = cplargs['nfvars']
    if sign == 'positive':
        op = 1.0
    elif sign == 'negative':
        op = -1.0
    else:
        raise ValueError("Wrong sign of viscous jacobian")

    def visjacobian(uf, nf, A, mu, rcp_dx):
        rho = uf[0]

        lam = op*rcp_dx/rho * max(4/3, gamma)*mu/pr

        for idx in range(nfvars):
            A[idx, idx] += lam

    return be.compile(visjacobian)


def make_none_jacobian(be, cplargs, sign):
    """
    No viscous flux Jacobian
    """

    def visjacobian(uf, nf, A, mu, rcp_dx):
        pass

    return be.compile(visjacobian)