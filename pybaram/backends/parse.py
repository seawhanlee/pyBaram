# -*- coding: utf-8 -*-
"""Source transformations used by the CPU and CUDA backends.

Solver kernels are written as regular Python functions with an outer
``range(i_begin, i_end)`` loop. This module extracts their source and closure
variables, then performs the small backend-specific transformations needed by
Numba. It is intentionally a constrained source rewriter, not a general
Python parser.
"""

from math import sqrt, tanh, exp
from numba import cuda
import numpy as np
import inspect
import re


_header_pattern = r'def\s(\w+)\(\s*([\w_]+)\s*,\s*([\w_]+)'

def parse_loop(func, src='none', parallel=False, *kwargs):
    """Prepare a loop kernel for CPU execution.

    The function source is dedented and its closure variables are returned as
    the globals used to recreate it. With ``parallel='cpu'``, the outer
    ``range(i_begin, i_end)`` loop is changed to ``numba.prange`` before the
    caller compiles the recreated function.

    :param func: Kernel function whose first two arguments delimit the outer
                 loop.
    :type func: object
    :param str src: Explicit function source, or ``'none'`` to inspect
                    ``func``.
    :param parallel: Use ``'cpu'`` to rewrite the outer loop for CPU
                     parallelism.
    :return: Transformed source, closure globals, and function name.
    :rtype: tuple
    """
    # Obtain source
    if src == 'none':
        ftxt = inspect.getsource(func).split('\n')
    else:
        ftxt = src.split('\n')

    # global variables from closure (non nocals and globals)
    closure = inspect.getclosurevars(func)
    gvars = {**closure.nonlocals, **closure.globals}

    # Strip text
    npad = len(ftxt[0]) - len(ftxt[0].lstrip())
    ftxt = [l[npad:] for l in ftxt]

    # Header
    header = ftxt[0]

    # get name and arguments
    m = re.match(_header_pattern, header)
    name = m.group(1)

    # compile regex pattern for main loop
    loop_pattern = r"\s+for\s+[\w_]+\s+in\s+range\({}\s*,\s*{}".format(
        m.group(2), m.group(3)
        )

    # find lines of main loop
    for i, l in enumerate(ftxt):
        if re.match(loop_pattern, l):
            break

    if parallel == 'cpu':
        # replase loop text
        ftxt[i] = re.sub(r"in\s+range\(", 'in nb.prange(', ftxt[i])

    # Rewrite header, loop and padded source
    ftxt ='\n'.join(ftxt)

    return ftxt, gvars, name


_header_pattern_simple = r'def\s(\w+)\('

# CUDA recognizable functions
_cuda_syms = {'array': 'cuda.local.array', 'np.sqrt': 'sqrt',
              'np.abs': 'abs', 'np.tanh': 'tanh', 'np.exp' : 'exp'}
_cuda_vars = {'sqrt': sqrt, 'abs': abs, 'tanh': tanh, 'exp' : exp}

def gpu_loop_begin(idxname, n0):
    """Return the CUDA grid-index prologue for a transformed loop kernel."""
    if n0 == 0:
        ftxt = (
            f"    {idxname} = cuda.grid(1)\n"
            f"    if {idxname} < i_end:"
        )
    else:
        ftxt = (
            f"    {idxname} = cuda.grid(1)\n"
            f"    if {idxname} < i_end - i_begin:\n"
            f"        {idxname} += i_begin"
        )
    return ftxt


def parse_loop_gpu(func, n0, src='none'):
    """Transform an outer-loop kernel into a one-dimensional CUDA kernel.

    The outer ``range(i_begin, i_end)`` loop is replaced by a
    ``cuda.grid(1)`` index and bounds check. Supported local-array and NumPy
    spellings are also replaced with CUDA-compatible symbols. The caller
    recreates the returned source and compiles it with ``cuda.jit``.

    :param func: Kernel function whose first two arguments delimit the outer
                 loop.
    :type func: object
    :param int n0: Compile-time lower bound used to construct the CUDA index.
    :param str src: Explicit function source, or ``'none'`` to inspect
                    ``func``.
    :return: Transformed source, closure globals, and function name.
    :rtype: tuple
    """
    if src == 'none':
        ftxt = inspect.getsource(func).split('\n')
    else:
        ftxt = src.split('\n')

    # global variables from closure (non nocals and globals)
    closure = inspect.getclosurevars(func)
    gvars = {**closure.nonlocals, **closure.globals}

    # Strip text
    npad = len(ftxt[0]) - len(ftxt[0].lstrip())
    ftxt = [l[npad:] for l in ftxt]

    # Header
    header = ftxt[0]

    # get name and arguments
    m = re.match(_header_pattern, header)
    name = m.group(1)

    # Compile regex pattern for main loop
    loop_pattern = r"\s+for\s+(\w+)+\s+in\s+range\({}\s*,\s*{}".format(
        m.group(2), m.group(3)
        )

    # Substitution for CUDA kernel
    for i, l in enumerate(ftxt):
        # Substitute Numpy expressions
        if any([re.search(key, l) for key in _cuda_syms.keys()]):
            gvars.update(_cuda_vars)
            for key, val in _cuda_syms.items():
                ftxt[i] = ftxt[i].replace(key, val, 3)

    # find lines of main loop
    for i, l in enumerate(ftxt):
        n = re.match(loop_pattern, l)
        if n:
            idxname = n.group(1)
            break

    ftxt[i] = gpu_loop_begin(idxname, n0)

    # Rewrite header, loop and padded source
    ftxt = '\n'.join(ftxt)

    return ftxt, gvars, name


def parse_simple_gpu(func, src='none'):
    """Prepare a loop-free helper function for CUDA device compilation.

    Unlike :func:`parse_loop_gpu`, this function does not generate a CUDA grid
    index. It only dedents the source, captures closure variables, and replaces
    supported local-array and NumPy spellings. ``GPUBackend`` uses the result
    to compile reusable device functions.

    :param func: Helper function to transform.
    :type func: object
    :param str src: Explicit function source, or ``'none'`` to inspect
                    ``func``.
    :return: Transformed source, closure globals, and function name.
    :rtype: tuple
    """
    if src == 'none':
        ftxt = inspect.getsource(func).split('\n')
    else:
        ftxt = src.split('\n')

    # global variables from closure (non nocals and globals)
    closure = inspect.getclosurevars(func)
    gvars = {**closure.nonlocals, **closure.globals}

    # Strip text
    npad = len(ftxt[0]) - len(ftxt[0].lstrip())
    ftxt = [l[npad:] for l in ftxt]

    # Header
    header = ftxt[0]

    # get name and arguments
    m = re.match(_header_pattern_simple, header)
    name = m.group(1)

    # Substitution for CUDA kernel
    for i, l in enumerate(ftxt):
        # Substitute Numpy expressions
        if any([re.search(key, l) for key in _cuda_syms.keys()]):
            gvars.update(_cuda_vars)
            for key, val in _cuda_syms.items():
                ftxt[i] = ftxt[i].replace(key, val, 3)

    ftxt = '\n'.join(ftxt)

    return ftxt, gvars, name
