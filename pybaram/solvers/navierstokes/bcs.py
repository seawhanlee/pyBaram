# -*- coding: utf-8 -*-
from pybaram.utils.nb import dot
from pybaram.solvers.euler.bcs import (make_bc_far, make_bc_sup_out, make_bc_sup_in,
                                       make_bc_sub_inv, make_bc_sub_outp, make_bc_slip_wall,
                                       make_bc_sub_inptt)


def get_bc(self, be, name, bcargs):
    bc = eval('make_bc_'+name)
    return be.compile(bc(bcargs))


def make_bc_adia_wall(bcargs):
    nvars, ndims = bcargs['nfvars'], bcargs['ndims']

    def bc(ul, ur, nf):
        ur[0] = ul[0]

        for idx in range(ndims):
            ur[idx+1] = -ul[idx+1]

        ur[nvars-1] = ul[nvars-1]

    return bc


def make_bc_isotherm_wall(bcargs):
    nvars, ndims = bcargs['nfvars'], bcargs['ndims']
    gamma = bcargs['gamma']
    pmin = bcargs['pmin']   
    cptw = bcargs['cptw']
    
    def bc(ul, ur, nf):
        # Specific Enthalpy
        p = max((gamma - 1)*(ul[nvars-1] - 0.5 *
                             dot(ul, ul, ndims, 1, 1)/ul[0]), pmin)
        
        # Compute wall enthalpy
        cptl = p/ul[0]/(gamma-1)*gamma
        cptr = 2*cptw - cptl

        ur[0] = gamma / (gamma-1)*p/cptr

        for idx in range(ndims):
            ur[idx+1] = -ul[idx+1]/ul[0]*ur[0]

        ur[nvars-1] = ur[0]*cptr/gamma + 0.5*dot(ur, ur, ndims, 1, 1) / ur[0]

    return bc
