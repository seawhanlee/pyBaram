# -*- coding: utf-8 -*-
from pybaram.backends.parse import parse_loop

import numba as nb
import math
            
            
def make_serial_loop1d(ne, func, n0=0, debug=False, src='none'):
    # Compile function
    if debug:
        # Don't JIT compile if debug mode
        return lambda *args : func(n0, ne, *args)
    else:
        # Compile serial loop
        _func = nb.jit(nopython=True, fastmath=True)(func)

        # Dispatch kernel
        def kern(*args):
            _func(n0, ne, *args)

        return kern
    

def make_parallel_loop1d(ne, func, n0=0, src='none'):
    # Parser to enable parallel loop
    ftxt, gvars, name = parse_loop(func, src, parallel='cpu')

    # Bind with global variable
    gvars.update({'n0': n0, 'ne': ne, 'nb' : nb})
    lvars = {}
    exec(ftxt, gvars, lvars)
    
    # Compile parallel loop
    _func = nb.jit(nopython=True, fastmath=True, parallel=True)(lvars[name])

    # Dispatch kernel
    def kern(*args):
        _func(n0, ne, *args)

    return kern

