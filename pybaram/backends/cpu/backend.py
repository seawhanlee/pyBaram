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

    def __init__(self, cfg, **kwargs):
        self.comp_stream = None
        self.copy_stream = None

        # Get mutli-thread type
        self.multithread = multithread = cfg.get('backend-cpu', 'multi-thread', default='single')

        # Loop structure for multi-thread type
        if multithread == 'single':
            self.make_loop = make_serial_loop1d
            self.cpu_workers = 1
            
            # Enforce to disable OpenMP
            os.environ['OMP_NUM_THREADS'] = '1'
        else:
            self.make_loop = make_parallel_loop1d

            # Threading layer selection
            if multithread in ['default', 'forksafe', 'threadsafe', 'safe', 'omp', 'tbb']:
                nb.config.THREADING_LAYER = multithread

            # Follow the actual Numba thread count.
            self.cpu_workers = nb.get_num_threads()

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

    def alloc_array(self, shape, dtype=np.float64, mapped=False, pinned=False, src=None, init=None):
        # Compatibility for GPU backend
        if mapped:
            arr = np.empty(shape, dtype)
            return arr, arr
        if src is not None:
            return src.astype(dtype, copy=True)      # return copy of original array
        
        if init is None:
            return np.empty(shape, dtype)
        else:
            return np.full(shape, init, dtype)
    
    def convert_array(self, array):
        # In CPU, return Numpy array itself
        return array
    
    def get_array(self, arrs, *args):
        # Return list of arrays
        return arrs
    
    def make_sum_reduce(self, nvars):
        def _run(array, reduced_array):
            reduced_array[:] = np.sum(array, axis=1)
        
        return _run

    def min_arrays(self, arrs):
        return arrs.min()

    def wait(self):
        # Dummy function
        pass

    def copy_array(self, *args):
        return np.copyto

    def wait_copy_stream(self):
        # Dummy function
        pass

    def sync_comp_to_copy(self):
        # Dummy function
        pass

    def sync_copy_to_comp(self):
        # Dummy function
        pass
