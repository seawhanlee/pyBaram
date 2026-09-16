# -*- coding: utf-8 -*-
from pybaram.backends import Backend
from pybaram.backends.cuda.loop import make_cuda_loop
from pybaram.backends.parse import parse_simple_gpu
from pybaram.utils.mpi import get_local_rank, get_local_size
from numba.core.errors import NumbaPerformanceWarning

from numba import cuda
import numpy as np
import os
import warnings


class GPUBackend(Backend):
    """
    Backend for GPU computation
    """
    name = 'cuda'

    def __init__(self, cfg, comm=None):
        # Assign GPU with local rank
        local_rank = get_local_rank(comm)
        ndev = len(cuda.gpus)
        if ndev < 1:
            raise RuntimeError('No CUDA device detected')
        cuda.select_device(local_rank % ndev)

        # Ignore GPU capability warning
        warnings.simplefilter("ignore", category=NumbaPerformanceWarning)

        # Read CUDA section first
        sect = 'backend-cuda'
        tpb = cfg.getint(sect, 'threads-per-block', 128)
        debug = eval(cfg.get(sect, 'debug', 'false').capitalize())

        local_size = max(1, get_local_size(comm))

        # Number of CPU workers per GPU.
        if cfg.has_option(sect, 'cpu-workers'):
            self.cpu_workers = cfg.getint(sect, 'cpu-workers', 1)
        else:
            try:
                ncpu = len(os.sched_getaffinity(0))
            except AttributeError:
                ncpu = os.cpu_count() or 1

            self.cpu_workers = max(1, ncpu // local_size)

        # Multiple streams for MPI communication
        if local_size == 1:
            self.streams = [cuda.stream()]
        else:
            self.streams = [cuda.stream() for _ in range(2)]

        self.comp_stream = self.streams[0]
        self.copy_stream = self.streams[min(1, len(self.streams) - 1)]

        self.make_loop = make_cuda_loop(self.comp_stream, tpb, debug)

        # Reduction function
        self._sum_reduce = cuda.reduce(lambda a, b : a + b)
        self._min_reduce = cuda.reduce(lambda a, b : min(a, b))

    def compile(self, func, src='none', **kwargs):
        ftxt, gvars, name = parse_simple_gpu(func, src)
        gvars.update({'cuda': cuda})
        lvars = {}
        exec(ftxt, gvars, lvars)

        return cuda.jit(device=True)(lvars[name])

    def local(self):
        # Dummy function
        # Returning `cuda.local.array` unavailable
        # Defined in `pybaram.backends.parse`
        return None

    def alloc_array(self, shape, dtype=np.float64, mapped=False, pinned=False, src=None, init=None):
        # Mapped (Zero-copy) memory for residual array
        if mapped:
            h_arr = cuda.mapped_array(shape, dtype=dtype)
            d_arr = cuda.as_cuda_array(h_arr)
            return h_arr, d_arr

        # Pinned memory for multiple streams (Host array)
        if pinned:
            return cuda.pinned_array(shape, dtype=dtype)

        # Copy source array
        if src is not None:
            return cuda.to_device(src.astype(dtype), copy=True)

        # Initialized array
        if init is None:
            return cuda.device_array(shape, dtype)
        else:
            return cuda.to_device(np.full(shape, init, dtype), copy=True)

    def convert_array(self, ndarray):
        # Return C contiguous GPU array
        if ndarray.flags['C_CONTIGUOUS']:
            return cuda.to_device(ndarray, stream=self.comp_stream)
        else:
            return cuda.to_device(np.ascontiguousarray(ndarray), stream=self.comp_stream)

    def get_array(self, d_arrs, h_arrs):
        # Copy device array to host array
        for d_arr, h_arr in zip(d_arrs, h_arrs):
            d_arr.copy_to_host(h_arr, stream=self.comp_stream)
        return h_arrs

    def make_sum_reduce(self, nvars):
        sum_reduce = self._sum_reduce
        def _run(array, reduced_array):
            for idx in range(nvars):
                sum_reduce(array[idx, :], res=reduced_array[idx:idx+1])

        return _run

    def min_arrays(self, arrs):
        return [self._min_reduce(arr, init=np.float64(np.inf)) for arr in arrs]

    def wait(self):
        # Host <-> Device synchronization
        cuda.synchronize()

    def copy_array(self, type, stream=None):
        stream = self.copy_stream if stream is None else stream

        def host_to_device(d_arr, h_arr):
            # with nvtx.annotate(name, color='purple'):
            d_arr.copy_to_device(h_arr, stream=stream)

        def device_to_host(h_arr, d_arr):
            # with nvtx.annotate(name, color='purple'):
            d_arr.copy_to_host(h_arr, stream=stream)

        def device_to_device(d_dest, d_src):
            # with nvtx.annotate(name, color='purple'):
            d_dest.copy_to_device(d_src, stream=stream)

        if type == 'h2d':
            return host_to_device
        elif type == 'd2h':
            return device_to_host
        elif type == 'd2d':
            return device_to_device
        else:
            raise ValueError("Invalid data transfer type")

    def wait_copy_stream(self):
        self.copy_stream.synchronize()

    def sync_comp_to_copy(self):
        if self.comp_stream != self.copy_stream:
            event = cuda.event(timing=False)
            event.record(self.comp_stream)
            event.wait(self.copy_stream)

    def sync_copy_to_comp(self):
        if self.comp_stream != self.copy_stream:
            event = cuda.event(timing=False)
            event.record(self.copy_stream)
            event.wait(self.comp_stream)
