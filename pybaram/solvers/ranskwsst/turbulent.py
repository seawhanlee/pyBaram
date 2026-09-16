# -*- coding: utf-8 -*-
import numpy as np


def make_blendingF1(be, cplargs):
    ndims, nvars = cplargs['ndims'], cplargs['nvars']
    betast = cplargs['betast']
    sigmaw2 = cplargs['sigmaw2']
    
    def f1(uc, gc, mu, d):
        rho = uc[0]
        k, w = uc[nvars-2]/rho, uc[nvars-1]/rho

        term1 = np.sqrt(k) / (betast*w*d)
        term2 = 500*mu/rho / (w*d**2)

        # Compute dk/dx_i dw/dx_i
        kwcross = 0
        for i in range(ndims):
            rho_x = gc[i][0]
            k_x = (gc[i][nvars-2] - k*rho_x)/rho
            w_x = (gc[i][nvars-1] - w*rho_x)/rho
            kwcross += k_x*w_x

        cdkw = max(2*rho*sigmaw2/w*kwcross, 1e-10)
        term3 = 4*rho*sigmaw2*k/cdkw/d**2

        arg1 = min(max(term1, term2), term3)

        return np.tanh(arg1**4)

    return be.compile(f1)


def make_blendingF2(be, cplargs):
    nvars = cplargs['nvars']
    betast = cplargs['betast']

    def f2(uc, mu, d):
        rho = uc[0]
        k, w = uc[nvars-2]/rho, uc[nvars-1]/rho

        term1 = (np.sqrt(k) / (betast*w*d))
        term2 = 500*mu/rho / (w*d**2)

        arg2 = max(2*term1, term2)

        return np.tanh(arg2**2)

    return be.compile(f2)
