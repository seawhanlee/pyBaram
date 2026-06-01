# -*- coding: utf-8 -*-
from pybaram.utils.nb import dot
from pybaram.utils.np import eps

import numpy as np
import re


def get_rsolver(name, be, cplargs):
    """
    docstring
    """
    fname = re.sub(r"\+", 'p', name)
    fname = re.sub('-', '_', fname)
    flux = eval('make_' + fname)(cplargs)

    return be.compile(flux)


def make_rusanov(cplargs):
    nvars, gamma = cplargs['nfvars'], cplargs['gamma']
    flux = cplargs['flux']
    array = cplargs['array']

    def rsolver(ul, ur, nf, fn):
        fl, fr = array((nvars,), np.float64), array((nvars,), np.float64)
        
        pl, contravl = flux(ul, nf, fl)
        pr, contravr = flux(ur, nf, fr)

        contrav = 0.5*(contravl + contravr)
        an = np.sqrt(gamma*(pl+pr) / (ul[0] + ur[0])) + np.abs(contrav)

        for jdx in range(nvars):
            fn[jdx] = 0.5*(fl[jdx] + fr[jdx]) - 0.5*an*(ur[jdx] - ul[jdx])

    return rsolver


def make_roe(cplargs):
    ndims, nvars, gamma = cplargs['ndims'], cplargs['nfvars'], cplargs['gamma']
    flux = cplargs['flux']
    array = cplargs['array']

    def rsolver(ul, ur, nf, fn):
        fl, fr = array((nvars,), np.float64), array((nvars,), np.float64)
        vl, vr = array((ndims,), np.float64), array((ndims,), np.float64)
        dv, va = array((ndims,), np.float64), array((ndims,), np.float64)
        #ev1, ev2, ev3 = array((nvars,)), array((nvars,)), array((nvars,))
        ev = array((3, nvars), np.float64)

        pl, contravl = flux(ul, nf, fl)
        pr, contravr = flux(ur, nf, fr)

        # Specific enthalpy, contra velocity for left / right
        for jdx in range(ndims):
            vl[jdx] = ul[jdx+1] / ul[0]
            vr[jdx] = ur[jdx+1] / ur[0]

        hl = (ul[nvars-1] + pl)/ul[0]
        hr = (ur[nvars-1] + pr)/ur[0]

        # Difference between two state
        drho = ur[0] - ul[0]
        dp = pr - pl
        dcontrav = contravr - contravl
        for jdx in range(ndims):
            dv[jdx] = vr[jdx] - vl[jdx]

        # Compute Roe averaged density and enthalpy
        rrr = np.sqrt(ur[0]/ul[0])
        ratl = 1.0/(1.0 + rrr)
        ratr = rrr*ratl
        ra = rrr*ul[0]
        ha = hl*ratl + hr*ratr

        for jdx in range(ndims):
            va[jdx] = vl[jdx]*ratl + vr[jdx]*ratr

        contrava = dot(va, nf, ndims, 0, 0)

        qq = 0.5*dot(va, va, ndims, 0, 0)
        aasq = (gamma - 1)*(ha - qq)
        aa = np.sqrt(aasq)
        inv_aasq = 1/aasq

        l1 = np.abs(contrava - aa)
        l2 = np.abs(contrava)
        l3 = np.abs(contrava + aa)

        # Entropy fix
        eps = 0.2
        if l1 < eps:
            l1 = 0.5*(l1**2/eps + eps)
        
        if l3 < eps:
            l3 = 0.5*(l3**2/eps + eps)

        alp1 = 0.5*(dp - ra*aa*dcontrav)*inv_aasq
        alp2 = drho - dp*inv_aasq
        alp3 = 0.5*(dp + ra*aa*dcontrav)*inv_aasq
        
        ev[0, 0] = alp1
        ev[0, nvars-1] = alp1*(ha - aa*contrava)

        ev[1, 0] = alp2
        ev[1, nvars-1] = alp2*qq + ra*(dot(va, dv, ndims, 0, 0) - contrava*dcontrav)

        ev[2, 0] = alp3
        ev[2, nvars-1] = alp3*(ha + aa*contrava)

        for jdx in range(ndims):
            ev[0, 1+jdx] = alp1*(va[jdx] - aa*nf[jdx])
            ev[1, 1+jdx] = alp2*va[jdx] + ra*(dv[jdx] - nf[jdx]*dcontrav)
            ev[2, 1+jdx] = alp3*(va[jdx] + aa*nf[jdx])

        for jdx in range(nvars):
            fn[jdx] = 0.5*(fl[jdx] + fr[jdx]) - 0.5*(l1*ev[0, jdx] + l2*ev[1, jdx] + l3*ev[2, jdx])

    return rsolver


def make_roem(cplargs):
    ndims, nvars, gamma = cplargs['ndims'], cplargs['nfvars'], cplargs['gamma']
    flux = cplargs['flux']
    array = cplargs['array']

    def rsolver(ul, ur, nf, fn):
        fl, fr = array((nvars,), np.float64), array((nvars,), np.float64)
        vl, vr = array((ndims,), np.float64), array((ndims,), np.float64)
        dv, va = array((ndims,), np.float64), array((ndims,), np.float64)
        du, bdq = array((nvars,), np.float64), array((nvars,), np.float64)

        pl, contravl = flux(ul, nf, fl)
        pr, contravr = flux(ur, nf, fr)

        # Specific enthalpy, contra velocity for left / right
        for jdx in range(ndims):
            vl[jdx] = ul[jdx+1] / ul[0]
            vr[jdx] = ur[jdx+1] / ur[0]

        hl = (ul[nvars-1] + pl)/ul[0]
        hr = (ur[nvars-1] + pr)/ur[0]

        # Difference between two state
        drho = ur[0] - ul[0]
        dp = pr - pl
        dh = hr - hl
        dcontrav = contravr - contravl
        for jdx in range(ndims):
            dv[jdx] = vr[jdx] - vl[jdx]

        # Compute Roe averaged density and enthalpy
        rrr = np.sqrt(ur[0]/ul[0])
        ratl = 1.0/(1.0 + rrr)
        ratr = rrr*ratl
        ra = rrr*ul[0]
        ha = hl*ratl + hr*ratr

        for jdx in range(ndims):
            va[jdx] = vl[jdx]*ratl + vr[jdx]*ratr

        contrava = dot(va, nf, ndims, 0, 0)

        aa = np.sqrt((gamma - 1)*(ha - 0.5*dot(va, va, ndims, 0, 0)))
        rcp_aa = 1/aa

        # Compute |M|, add a small number to avoid a possible singularity of f
        abs_ma = np.abs(contrava*rcp_aa) + eps

        # Eigen structure
        b1 = max(0.0, max(contrava + aa, contravr + aa))
        b2 = min(0.0, min(contrava - aa, contravl - aa))

        #  Normalized wave speed
        b1b2 = b1*b2
        rcp_b1_b2 = 1.0/(b1 - b2)
        b1 = b1*rcp_b1_b2
        b2 = b2*rcp_b1_b2
        b1b2 = b1b2*rcp_b1_b2

        # 1-D shock discontinuity sensing term and Mach number based function f,g
        if pl < pr:
            SDST = pl / pr
        else:
            SDST = pr / pl

        h = 1.0 - SDST
        f = abs_ma**h
        g = f/(1.0 + abs_ma)

        for jdx in range(nvars-1):
            du[jdx] = ur[jdx] - ul[jdx]
        du[nvars-1] = ur[0]*hr - ul[0]*hl

        # BdQ
        bdq[0] = drho - f*dp*rcp_aa**2
        bdq[nvars - 1] = bdq[0]*ha + ra*dh
        for jdx in range(ndims):
            bdq[jdx+1] = bdq[0]*va[jdx] + ra*(dv[jdx] - nf[jdx]*dcontrav)

        for jdx in range(nvars):
            fn[jdx] = b1*fl[jdx] - b2*fr[jdx] + b1b2*(du[jdx] - g*bdq[jdx])

    return rsolver


def make_rotated_roem(cplargs):
    ndims, nvars, gamma = cplargs['ndims'], cplargs['nfvars'], cplargs['gamma']
    flux = cplargs['flux']
    array = cplargs['array']

    def rsolver(ul, ur, nf, fn):
        fl, fr = array((nvars,), np.float64), array((nvars,), np.float64)
        vl, vr = array((ndims,), np.float64), array((ndims,), np.float64)
        dv, va = array((ndims,), np.float64), array((ndims,), np.float64)
        du, bdq = array((nvars,), np.float64), array((nvars,), np.float64)
        nv = array((ndims,), np.float64)

        pl, contravl = flux(ul, nf, fl)
        pr, contravr = flux(ur, nf, fr)

        # Specific enthalpy, contra velocity for left / right
        for jdx in range(ndims):
            vl[jdx] = ul[jdx+1] / ul[0]
            vr[jdx] = ur[jdx+1] / ur[0]

        hl = (ul[nvars-1] + pl)/ul[0]
        hr = (ur[nvars-1] + pr)/ur[0]

        # Difference between two state
        drho = ur[0] - ul[0]
        dp = pr - pl
        dh = hr - hl
        for jdx in range(ndims):
            dv[jdx] = vr[jdx] - vl[jdx]

        # Compute Roe averaged density and enthalpy
        rrr = np.sqrt(ur[0]/ul[0])
        ratl = 1.0/(1.0 + rrr)
        ratr = rrr*ratl
        ra = rrr*ul[0]
        ha = hl*ratl + hr*ratr

        for jdx in range(ndims):
            va[jdx] = vl[jdx]*ratl + vr[jdx]*ratr

        contrava = dot(va, nf, ndims, 0, 0)

        aa = np.sqrt((gamma - 1)*(ha - 0.5*dot(va, va, ndims, 0, 0)))
        rcp_aa = 1/aa

        # Eigen structure
        b1 = max(0.0, max(contrava + aa, contravr + aa))
        b2 = min(0.0, min(contrava - aa, contravl - aa))

        #  Normalized wave speed
        b1b2 = b1*b2
        rcp_b1_b2 = 1.0/(b1 - b2)
        b1 = b1*rcp_b1_b2
        b2 = b2*rcp_b1_b2
        b1b2 = b1b2*rcp_b1_b2

        # Rotated direction       
        mag_dv = np.sqrt(dot(dv, dv, ndims, 0, 0))
        if mag_dv < 1e-6:
            # For very small dv
            for jdx in range(ndims):
                nv[jdx] = nf[jdx]
        else:
            # Velocity direction
            for jdx in range(ndims):
                nv[jdx] = (dv[jdx] / mag_dv)

            # Directional Mach based function beta
            abs_ma1 = abs(contrava)*rcp_aa
            beta = 1 - np.exp(-50*abs_ma1)
            
            # Averaging nv and nf with beta 
            mag_nv = 0
            for jdx in range(ndims):
                nv[jdx] = beta*nv[jdx] + (1-beta)*nf[jdx]
                mag_nv += nv[jdx]**2

            # Unit vector
            mag_nv = np.sqrt(mag_nv)
            for jdx in range(ndims):
                nv[jdx] /= mag_nv

        # Directional Cosine
        alp = dot(nv, nf, ndims, 0, 0)

        # Rotated Contact wave terms
        contravav = dot(va, nv, ndims, 0, 0)
        contravlv = dot(vl, nv, ndims, 0, 0)
        contravrv = dot(vr, nv, ndims, 0, 0)
        dcontrav = contravrv - contravlv

        b1v = max(0.0, max(contravav + aa, contravrv + aa))
        b2v = min(0.0, min(contravav - aa, contravlv - aa))
        b1b2v = b1v*b2v / (b1v - b2v)

        # Enforce upwind for supersonic
        if b1b2 == 0:
            b1b2v = 0

        abs_ma = np.abs(contravav*rcp_aa)
        rcp_aba_ma_p1 = 1 /(1.0 + abs_ma)

        for jdx in range(nvars-1):
            du[jdx] = ur[jdx] - ul[jdx]
        du[nvars-1] = ur[0]*hr - ul[0]*hl

        # BdQ
        bdq[0] = drho - abs(alp)*dp*rcp_aa**2
        bdq[nvars - 1] = bdq[0]*ha + ra*dh
        for jdx in range(ndims):
            bdq[jdx+1] = bdq[0]*va[jdx] + ra*(dv[jdx] - nv[jdx]*dcontrav)

        for jdx in range(nvars):
            fn[jdx] = b1*fl[jdx] - b2*fr[jdx] + b1b2*du[jdx] - alp*b1b2v*rcp_aba_ma_p1*bdq[jdx]

    return rsolver


def make_hlle(cplargs):
    ndims, nvars, gamma = cplargs['ndims'], cplargs['nfvars'], cplargs['gamma']
    flux = cplargs['flux']
    array = cplargs['array']

    def rsolver(ul, ur, nf, fn):
        fl, fr = array((nvars,), np.float64), array((nvars,), np.float64)
        vl, vr = array((ndims,), np.float64), array((ndims,), np.float64)
        va = array((ndims,), np.float64)

        pl, contravl = flux(ul, nf, fl)
        pr, contravr = flux(ur, nf, fr)

        # Specific enthalpy, contra velocity for left / right
        for jdx in range(ndims):
            vl[jdx] = ul[jdx+1] / ul[0]
            vr[jdx] = ur[jdx+1] / ur[0]

        hl = (ul[nvars-1] + pl)/ul[0]
        hr = (ur[nvars-1] + pr)/ur[0]

        # Compute Roe averaged density and enthalpy
        rrr = np.sqrt(ur[0]/ul[0])
        ratl = 1.0/(1.0 + rrr)
        ratr = rrr*ratl
        ha = hl*ratl + hr*ratr

        for jdx in range(ndims):
            va[jdx] = vl[jdx]*ratl + vr[jdx]*ratr

        contrava = dot(va, nf, ndims, 0, 0)
        aa = np.sqrt((gamma - 1)*(ha - 0.5*dot(va, va, ndims, 0, 0)))

        # Eigen structure
        b1 = max(0.0, max(contrava + aa, contravr + aa))
        b2 = min(0.0, min(contrava - aa, contravl - aa))

        #  Normalized wave speed
        b1b2 = b1*b2
        rcp_b1_b2 = 1.0/(b1 - b2)
        b1 = b1*rcp_b1_b2
        b2 = b2*rcp_b1_b2
        b1b2 = b1b2*rcp_b1_b2

        for jdx in range(nvars):
            fn[jdx] = b1*fl[jdx] - b2*fr[jdx] \
                + b1b2*(ur[jdx] - ul[jdx])

    return rsolver


def make_hllem(cplargs):
    ndims, nvars, gamma = cplargs['ndims'], cplargs['nfvars'], cplargs['gamma']
    flux = cplargs['flux']
    array = cplargs['array']

    def rsolver(ul, ur, nf, fn):
        fl, fr = array((nvars,), np.float64), array((nvars,), np.float64)
        vl, vr = array((ndims,), np.float64), array((ndims,), np.float64)
        dv, va = array((ndims,), np.float64), array((ndims,), np.float64)
        df = array((ndims,), np.float64)

        pl, contravl = flux(ul, nf, fl)
        pr, contravr = flux(ur, nf, fr)

        # Specific enthalpy, contra velocity for left / right
        for jdx in range(ndims):
            vl[jdx] = ul[jdx+1] / ul[0]
            vr[jdx] = ur[jdx+1] / ur[0]

        hl = (ul[nvars-1] + pl)/ul[0]
        hr = (ur[nvars-1] + pr)/ur[0]

        # Compute Roe averaged density and enthalpy
        rrr = np.sqrt(ur[0]/ul[0])
        ratl = 1.0/(1.0 + rrr)
        ratr = rrr*ratl
        ra = rrr*ul[0]
        ha = hl*ratl + hr*ratr

        for jdx in range(ndims):
            va[jdx] = vl[jdx]*ratl + vr[jdx]*ratr

        qq = dot(va, va, ndims, 0, 0)
        contrava = dot(va, nf, ndims, 0, 0)
        aa = np.sqrt((gamma - 1)*(ha - 0.5*dot(va, va, ndims, 0, 0)))

        # Eigen structure
        b1 = max(0.0, max(contrava + aa, contravr + aa))
        b2 = min(0.0, min(contrava - aa, contravl - aa))

        # Contact term
        um = 0.5*(b1 + b2)
        delta = aa / (aa + abs(um))

        # Difference between two states
        drho = ur[0] - ul[0]
        dp = pr - pl
        dcontra = contravr - contravl

        for jdx in range(ndims):
            dv[jdx] = vr[jdx] - vl[jdx]

        alp1 = drho - dp/aa**2

        df[0] = alp1
        for jdx in range(ndims):
            df[jdx+1] = alp1*va[jdx]
        df[nvars-1] = 0.5*alp1*qq

        alp2 = ra
        for jdx in range(ndims):
            df[jdx+1] += alp2*(dv[jdx] - nf[jdx]*dcontra)
        df[nvars-1] += alp2*(dot(va, dv, ndims, 0, 0) - contrava*dcontra)

        #  Normalized wave speed
        b1b2 = b1*b2
        rcp_b1_b2 = 1.0/(b1 - b2)
        b1 = b1*rcp_b1_b2
        b2 = b2*rcp_b1_b2
        b1b2 = b1b2*rcp_b1_b2

        for jdx in range(nvars):
            fn[jdx] = b1*fl[jdx] - b2*fr[jdx] \
                + b1b2*((ur[jdx] - ul[jdx]) - delta*df[jdx])

    return rsolver


def make_ausmpwp(cplargs):
    array = cplargs['array']
    ndims, nvars, gamma = cplargs['ndims'], cplargs['nfvars'], cplargs['gamma']
    to_primevars = cplargs['to_primevars']

    alpha = 3/16

    def rsolver(ul, ur, nf, fn):
        vl, vr = array((ndims,), np.float64), array((ndims,), np.float64)
        pl = to_primevars(ul, vl)
        pr = to_primevars(ur, vr)

        # Specific enthalpy, contra velocity, tangential velocity for left / right
        hl = (ul[nvars-1] + pl)/ul[0]
        hr = (ur[nvars-1] + pr)/ur[0]

        contral = dot(vl, nf, ndims, 0, 0)
        contrar = dot(vr, nf, ndims, 0, 0)
        vl2 = dot(vl, vl, ndims, 0, 0) - contral**2
        vr2 = dot(vr, vr, ndims, 0, 0) - contrar**2

        conmid = 0.5*(contral + contrar)

        # Specific enthalpy along normal
        hn = 0.5*(hl - 0.5*vl2 + hr - 0.5*vr2)
        cs2 = 2.0*(gamma - 1)/(gamma + 1)*hn
        cs = np.sqrt(cs2)

        # Speed of sound at midpoint, mach number
        if conmid > 0:
            cmid = cs2/max(abs(contral), cs)
        else:
            cmid = cs2/max(abs(contrar), cs)

        ml = contral/cmid
        mr = contrar/cmid

        # AUSM-type function
        if abs(ml) < 1.0:
            mlp = 0.25*(ml + 1.0)**2
            plp = mlp*(2.0 - ml) + alpha*ml*(ml**2 - 1.0)**2
        else:
            mlp = 0.5*(ml + abs(ml))
            plp = mlp / ml

        if abs(mr) < 1.0:
            mrm = -0.25*(mr - 1.0)**2
            prm = -mrm*(2.0 + mr) - alpha*mr*(mr**2 - 1.0)**2
        else:
            mrm = 0.5*(mr - abs(mr))
            prm = mrm / mr

        m_mid = 0.5*(mlp + mrm)
        p_mid = plp*pl + prm*pr

        # 1-D shock discontinuity sensing term and pressure based function f, w
        if pl < pr:
            SDST = pl / pr
        else:
            SDST = pr / pl

        SDST2 = SDST**2

        if p_mid > eps:
            fl = (pl/p_mid - 1.0)*SDST2
            fr = (pr/p_mid - 1.0)*SDST2
        else:
            fl = 0
            fr = 0

        wei = 1.0 - SDST*SDST2

        # M+. M-
        if m_mid > 0.0:
            mp = mlp + mrm*((1.0 - wei)*(1.0 + fr) - fl)
            mm = mrm*wei*(1.0 + fr)
        else:
            mp = mlp*wei*(1.0 + fl)
            mm = mrm + mlp*((1.0 - wei)*(1.0 + fl) - fr)

        # Flux
        for jdx in range(nvars - 1):
            fn[jdx] = cmid*(mp*ul[jdx] + mm*ur[jdx])

        for jdx in range(ndims):
            fn[jdx + 1] += nf[jdx] * p_mid

        fn[nvars - 1] = cmid*(mp*ul[0]*hl + mm*ur[0]*hr)

    return rsolver


def make_ausmpup(cplargs):
    array = cplargs['array']
    ndims, nvars, gamma = cplargs['ndims'], cplargs['nfvars'], cplargs['gamma']
    to_primevars = cplargs['to_primevars']

    alpha, beta = 3/16, 1/8
    kp, ku = 1, 1

    def rsolver(ul, ur, nf, fn):
        vl, vr = array((ndims,), np.float64), array((ndims,), np.float64)
        pl = to_primevars(ul, vl)
        pr = to_primevars(ur, vr)

        # Specific enthalpy, contra velocity, tangential velocity for left / right
        hl = (ul[nvars-1] + pl)/ul[0]
        hr = (ur[nvars-1] + pr)/ur[0]

        contral = dot(vl, nf, ndims, 0, 0)
        contrar = dot(vr, nf, ndims, 0, 0)

        # Critical speed of sound
        csl2 = 2.0*(gamma - 1)/(gamma + 1)*hl
        csr2 = 2.0*(gamma - 1)/(gamma + 1)*hr

        ccl = csl2/max(contral, np.sqrt(csl2))
        ccr = csr2/max(-contrar, np.sqrt(csr2))

        # Speed of sound at midpoint, mach number
        rhmid = 0.5*(ul[0] + ur[0])
        cmid = min(ccl, ccr)
        ml = contral/cmid
        mr = contrar/cmid

        # AUSM-type function
        if abs(ml) < 1.0:
            mlp = 0.25*(ml + 1.0)**2
            plp = mlp*(2.0 - ml) + alpha*ml*(ml**2 - 1.0)**2
            mlp += beta*(ml**2 - 1.0)**2
        else:
            mlp = 0.5*(ml + abs(ml))
            plp = mlp / ml

        if abs(mr) < 1.0:
            mrm = -0.25*(mr - 1.0)**2
            prm = -mrm*(2.0 + mr) - alpha*mr*(mr**2 - 1.0)**2
            mrm -= beta*(mr**2 - 1.0)**2
        else:
            mrm = 0.5*(mr - abs(mr))
            prm = mrm / mr

        # M+. M-
        mdp = 0.5*(ml**2 + mr**2)
        mdp = -kp*max(1 - mdp, 0.0)*(pr - pl) / (rhmid*cmid**2)
        mmid = mlp + mrm + mdp
        mp = 0.5*(mmid + abs(mmid))
        mm = 0.5*(mmid - abs(mmid))

        pu = -ku*plp*abs(prm)*rhmid*cmid*(contrar - contral)
        p_mid = plp*pl + prm*pr + pu

        # Flux
        for jdx in range(nvars - 1):
            fn[jdx] = cmid*(mp*ul[jdx] + mm*ur[jdx])

        for jdx in range(ndims):
            fn[jdx + 1] += nf[jdx] * p_mid

        fn[nvars - 1] = cmid*(mp*ul[0]*hl + mm*ur[0]*hr)

    return rsolver


def make_ausmzc(cplargs):
    array = cplargs['array']
    ndims, nvars, gamma = cplargs['ndims'], cplargs['nfvars'], cplargs['gamma']
    to_primevars = cplargs['to_primevars']

    alpha = 0.0

    def rsolver(ul, ur, nf, fn):
        vl, vr = array((ndims,), np.float64), array((ndims,), np.float64)
        pl = to_primevars(ul, vl)
        pr = to_primevars(ur, vr)

        vtl = np.sqrt(dot(vl, vl, ndims, 0, 0) + eps)
        vtr = np.sqrt(dot(vr, vr, ndims, 0, 0) + eps)

        cl = np.sqrt(gamma*pl / ul[0])
        cr = np.sqrt(gamma*pr / ur[0])

        # Speed of sound at midpoint, mach number (total)
        mtl = vtl / cl
        mtr = vtr / cr
        mtm = 0.5*(mtl + mtr)

        # Weighted average
        k1, k2 = 15, 0.45
        z = 0.5*(np.tanh(k1*(mtm-k2))+1.0)
        fc = 1 - (1 - min(ul[0]/ur[0], ur[0]/ul[0])**3)*min(pl/pr,pr/pl)**3
        z *= fc

        # Specific enthalpy and contra velocity for left / right
        hl = (ul[nvars-1] + pl)/ul[0]
        hr = (ur[nvars-1] + pr)/ur[0]

        contral = dot(vl, nf, ndims, 0, 0)
        contrar = dot(vr, nf, ndims, 0, 0)

        # Critical speed of sound
        csl2 = 2.0*(gamma - 1)/(gamma + 1)*hl
        csr2 = 2.0*(gamma - 1)/(gamma + 1)*hr

        ccl = csl2/max(abs(contral), np.sqrt(csl2))
        ccr = csr2/max(abs(contrar), np.sqrt(csr2))

        # Speed of sound at midpoint, mach number
        cmid = 0.5*(ccl + ccr)

        ml = contral/cmid
        mr = contrar/cmid

        # AUSM-type function
        if abs(ml) < 1.0:
            mlp = 0.25*(ml + 1.0)**2
            plp = mlp*(2.0 - ml) + alpha*ml*(ml**2 - 1.0)**2
            #mlp += beta*(ml**2 - 1.0)**2
        else:
            mlp = 0.5*(ml + abs(ml))
            plp = mlp / ml

        if abs(mr) < 1.0:
            mrm = -0.25*(mr - 1.0)**2
            prm = -mrm*(2.0 + mr) - alpha*mr*(mr**2 - 1.0)**2
            #mrm -= beta*(mr**2 - 1.0)**2
        else:
            mrm = 0.5*(mr - abs(mr))
            prm = mrm / mr

        m_mid = (mlp + mrm)
        p_mid = plp*pl + prm*pr

        # M+. M-
        if m_mid > 0.0:
            mp = mlp + (1-z)*mrm
            mm = z*mrm
        else:
            mp = z*mlp
            mm = (1-z)*mlp + mrm

        # Flux
        for jdx in range(nvars - 1):
            fn[jdx] = cmid*(mp*ul[jdx] + mm*ur[jdx])

        for jdx in range(ndims):
            fn[jdx + 1] += nf[jdx] * p_mid

        fn[nvars - 1] = cmid*(mp*ul[0]*hl + mm*ur[0]*hr)

    return rsolver
