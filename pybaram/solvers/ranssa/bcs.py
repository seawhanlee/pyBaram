# -*- coding: utf-8 -*-
from pybaram.solvers.euler.bcs import (make_bc_far, make_bc_sup_out, make_bc_sup_in,
                                       make_bc_sub_inv, make_bc_sub_outp, make_bc_slip_wall,
                                       make_bc_sub_inptt)

from pybaram.solvers.navierstokes.bcs import make_bc_adia_wall, make_bc_isotherm_wall


def get_bc(self, be, name, bcargs):
    cbc = be.compile(eval('make_bc_'+name)(bcargs))

    if name in ['adia_wall', 'isotherm_wall']:
        tbc = be.compile(make_turb_bc_wall(bcargs))
    elif name in ['far', 'sub_inv', 'sup_in']:
        tbc = be.compile(make_turb_bc_far(bcargs))
    else:
        tbc = be.compile(make_turb_bc_ext(bcargs))

    def bc(ul, ur, nf, mu, d1):
        cbc(ul, ur, nf)
        tbc(ul, ur, nf, mu, d1)

    return be.compile(bc)


def make_turb_bc_wall(bcargs):
    nvars = bcargs['nvars']

    def bc(ul, ur, nf, mu, d1):
        ur[nvars-1] = -ul[nvars-1]

    return bc


def make_turb_bc_ext(bcargs):
    nvars = bcargs['nvars']

    def bc(ul, ur, nf, mu, d1):
        ur[nvars-1] = ul[nvars-1]

    return bc


def make_turb_bc_far(bcargs):
    nvars = bcargs['nvars']
    nut = bcargs['nut']

    def bc(ul, ur, nf, mu, d1):
        ur[nvars-1] = nut

    return bc
