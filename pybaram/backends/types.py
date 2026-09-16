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


class MPIPackKernel:
    """
    Prepare an MPI send buffer.

    Runs the data pack kernel on the compute stream, then optionally copies the
    packed device buffer to the host send buffer on the copy stream.
    """
    def __init__(self, be, pack, dtoh=NullKernel()):
        self.be = be
        self.pack = pack
        self.dtoh = dtoh

    def __call__(self, *args):
        self.pack(*args)
        self.be.sync_comp_to_copy()
        self.dtoh()


class MPIUnpackKernel:
    """
    Reflect an MPI receive buffer into backend-local storage.

    For CUDA this first copies the host receive buffer to the device receive
    buffer, then runs any data unpack kernel after the copy is visible.
    """
    def __init__(self, be, htod=NullKernel(), unpack=NullKernel()):
        self.be = be
        self.htod = htod
        self.unpack = unpack

    def __call__(self, *args):
        self.htod()
        self.be.sync_copy_to_comp()
        self.unpack(*args)


class MPISendKernel:
    """
    Start an MPI send after the backend copy stream has made the host buffer
    visible to MPI.
    """
    def __init__(self, be, send):
        self.be = be
        self.send = send

    def __call__(self, *args):
        self.be.wait_copy_stream()
        return self.send(*args)


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
