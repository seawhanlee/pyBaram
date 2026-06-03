# -*- coding: utf-8 -*-
from mpi4py import MPI


def _extract(arg):
    """
    Parse argument for array bank
    """
    try:
        return arg.value
    except AttributeError:
        pass

    if isinstance(arg, tuple) and hasattr(arg[0], 'value'):
        # Parse tuple of array bank
        return tuple(e.value for e in arg)

    return arg
    

class ArrayBank:
    """
    ArrayBank object

    It stores list of arrays and point one of them.
    """
    def __init__(self, mat, idx):
        # Curren index
        self.idx = idx

        # Bank of array
        self.mat = mat

    @property
    def value(self):
        # Return current array in the bank
        return self.mat[self.idx]
    
    def set(self, v):
        self.mat[self.idx][:] = v


class NullKernel:
    def __call__(self, *args):
        pass


class Kernel:
    """
    Kernel object

    Stores static arguments and executes a function with
    both static and runtime arguments.
    """
    def __init__(self, fun, *args):
        self._fun = fun
        self._args = args

    def __call__(self, *args):
        # Merge static argument and dynamic argument
        combined = self._args + args

        # Parse args for Array bank object
        parsed = [_extract(arg) for arg in combined]

        # Run function
        return self._fun(*parsed)

    def update_args(self, *args):
        # Update static argument
        self._args = args

    @property
    def is_compiled(self):
        # Check the function is already JIT compiled or not
        return self._fun.signatures != []


class MetaKernel:
    """
    Meta kernel object

    It stores series of kernels and run all them.
    """
    def __init__(self, kerns):
        # Store series of kernels
        self._kerns = kerns
    
    def __call__(self, *args):
        # Run all kernel squentially
        for kern in self._kerns:
            kern.__call__(*args)


class Queue:
    """
    Simple Queue

    It collects MPI requests and synchronizes all these commnunications.
    """
    def __init__(self):
        self._reqs = []

    def sync(self):
        # Fire-off the stacked requests in the queue
        MPI.Prequest.Waitall(self._reqs)
        self._reqs = []

    def register(self, *reqs):
        # Stack mpi requests
        for req in reqs:
            self._reqs.append(req)
