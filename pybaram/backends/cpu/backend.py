# -*- coding: utf-8 -*-
from pybaram.backends import Backend
from pybaram.backends.cpu.loop import make_serial_loop1d, make_parallel_loop1d
from pybaram.backends.cpu.local import stack_empty
from numba.extending import register_jitable

import numba as nb
import numpy as np
import os


class CPUBackend(Backend):
    """
    Backend for CPU computation
    - Support single thread and multi threads
    - Just-in Time compile via Numba
    """
    name = 'cpu'

    def __init__(self, cfg):
        # Get mutli-thread type
        self.multithread = multithread = cfg.get('backend-cpu', 'multi-thread', default='single')

        # Loop structure for multi-thread type
        if multithread == 'single':
            self.make_loop = make_serial_loop1d
            
            # Enforce to disable OpenMP
            os.environ['OMP_NUM_THREADS'] = '1'
        else:
            self.make_loop = make_parallel_loop1d

            # Threading layer selection
            if multithread in ['default', 'forksafe', 'threadsafe', 'safe', 'omp', 'tbb']:
                nb.config.THREADING_LAYER = multithread

        self.reduction = np.sum     # Summation reduction
    
    def compile(self, func, outer=False, **kwargs):
        # JIT compile the Python function
        if self.multithread == 'single' or not outer:
            return nb.jit(nopython=True, fastmath=True)(func)
        else:
            # Enable Numba parallelization if the function is not nested
            return nb.jit(nopython=True, fastmath=True, parallel=True)(func)
    
    def local(self):
        np_dtype = np.float64

        @register_jitable
        def _array(shape, dtype=np_dtype):
            # Compute size of shape
            size = 1
            for i in range(len(shape)):
                size *= shape[i]

            arr = stack_empty(size, shape, dtype=dtype)
            return arr
        
        return _array

    def alloc_array(self, shape, dtype=np.float64, init=None, src=None, mapped=False):
        # Compatibility for GPU backend
        if mapped:
            arr = np.empty(shape, dtype)
            return arr, arr
        if src is not None:
            return src.astype(dtype, copy=True)      # return copy of original array
        
        if init is None:
            return np.empty(shape, dtype)
        elif init == 1:
            return np.ones(shape, dtype)
        elif init == 0:
            return np.zeros(shape, dtype)
        else:
            return np.full(shape, init, dtype)
    
    def convert_array(self, array):
        # In CPU, return Numpy array itself
        return array
    
    def get_array(self, arrs, *args):
        # Return list of arrays
        return arrs
    
    def reduce_array(self, nvars):
        def _run(array, reduced_array):
            reduced_array[:] = np.sum(array, axis=1)
        
        return _run

    def wait(self):
        # Dummy function
        pass