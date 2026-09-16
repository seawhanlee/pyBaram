# -*- coding: utf-8 -*-
from pybaram.backends.parse import parse_loop_gpu, parse_loop

from numba import cuda
import numba as nb
import inspect


def make_cuda_loop(default_stream, tpb, debug=False):
    if debug:
        import nvtx

    def _cuda_loop(ne, func, *carrs, n0=0, src='none', host=False, stream=None):
        # Return CPU function by force
        # `BaseInters.compute_dxc`
        if host:
            ftxt, gvars, name = parse_loop(func, src=src)
            # Bind with global variable
            gvars.update({'n0': n0, 'ne': ne})
            lvars = {}
            exec(ftxt, gvars, lvars)

            # Compile parallel loop
            _func = nb.jit(nopython=True, fastmath=True, parallel=True)(lvars[name])

            # Dispatch kernel
            def kern_cpu(*args):
                _func(n0, ne, *args)

            return kern_cpu, *carrs

        # Parser to enable CUDA loop
        ftxt, gvars, name = parse_loop_gpu(func, n0, src=src)

        # Bind with global variable
        gvars.update({'n0': n0, 'ne': ne, 'cuda': cuda})
        lvars = {}
        exec(ftxt, gvars, lvars)

        # Compile
        bpg = (ne - n0 + tpb - 1) // tpb
        _func = cuda.jit(lvars[name])
        stream = default_stream if stream is None else stream

        # Profile properties
        if debug:
            if inspect.ismethod(func) and func.__self__ is not None:
                fname = func.__qualname__
                fcls = func.__self__.__class__.__name__
            else:
                qname = func.__qualname__
                parts = qname.split('.')
                fcls = parts[0]
                fname = parts[-1]

            if fcls.endswith('Elements'):
                color = 'blue'
            elif fcls.endswith('Inters'):
                color = 'green'
            else:
                color = 'orange'

            def debug_kern(*args):
                with nvtx.annotate(fname, color=color):
                    _func[bpg, tpb, stream](n0, ne, *args)

            return debug_kern, *carrs

        # Normal execution
        else:
            def kern(*args):
                _func[bpg, tpb, stream](n0, ne, *args)

            return kern, *carrs

    return _cuda_loop
