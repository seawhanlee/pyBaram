# -*- coding: utf-8 -*-
# Original code
# https://github.com/PyFR/PyFR/blob/develop/pyfr/partitioners/metis.py
# Modified by jspark
# 
from ctypes import POINTER, c_void_p, c_int, c_int64

from pybaram.utils.ctypes import load_lib

import numpy as np


class METISWrapper:
    METIS_NOPTIONS = 40

    def __init__(self):
        # Load metis library
        lib = load_lib('metis')

        # Probe data types
        self._probe_types(lib)

        # Assign Metis functions
        self.METIS_SetDefaultOptions = lib.METIS_SetDefaultOptions
        self.METIS_SetDefaultOptions.argtypes = [c_void_p]

        self.METIS_PartGraphKway = lib.METIS_PartGraphKway
        self.METIS_PartGraphKway.argtypes = [
            POINTER(self.metis_int), POINTER(self.metis_int), c_void_p,
            c_void_p, c_void_p, c_void_p, c_void_p, POINTER(self.metis_int),
            c_void_p, c_void_p, c_void_p, POINTER(self.metis_int), c_void_p
        ]

    def _probe_types(self, lib):
        # Find integer type
        lib.METIS_SetDefaultOptions.argtypes = [c_void_p]
        opts = np.arange(0, 40, dtype=np.int64)
        err = lib.METIS_SetDefaultOptions(opts.ctypes)

        if opts[-1] != self.METIS_NOPTIONS - 1:
            self.metis_int = metis_int = c_int64
            self.metis_int_np = metis_int_np = np.int64
        else:
            self.metis_int = metis_int = c_int
            self.metis_int_np = metis_int_np = np.int32

        # Sample of part graph to find float
        opts = np.empty(40, dtype=metis_int_np)
        err = lib.METIS_SetDefaultOptions(opts.ctypes)

        xadj = np.array([0, 1, 3, 5, 6], dtype=metis_int_np)
        adjncy = np.array([1, 0, 2, 1, 3, 2], dtype=metis_int_np)

        nvtxs = metis_int(4)
        ncon = metis_int(1)
        nparts = metis_int(2)
        objval = metis_int()
        part = np.zeros(4, dtype=metis_int_np)
        tpwgts = np.ones(2, dtype=np.float32)
        tpwgts /= np.sum(tpwgts)

        lib.METIS_PartGraphKway.argtypes = [
            POINTER(metis_int), POINTER(metis_int), c_void_p, c_void_p,
            c_void_p, c_void_p, c_void_p, POINTER(metis_int), c_void_p,
            c_void_p, c_void_p, POINTER(metis_int), c_void_p]

        err = lib.METIS_PartGraphKway(
            nvtxs, ncon, xadj.ctypes, adjncy.ctypes, None, None, None,
            nparts, tpwgts.ctypes, None, opts.ctypes, objval, part.ctypes
        )

        if err == 1:
            self.metis_float_np = np.float32
        else:
            self.metis_float_np = np.float64

    def part_graph(self, nparts, nvtxs, xadj, adjncy, ncon=1, vwts=None,
                   vsize=None, adjwgt=None, opts=None, tpwgts=None,
                   ubvec=None):
        # Metis int type
        metis_int, metis_int_np = self.metis_int, self.metis_int_np

        # Convert integer inputs
        _nparts = metis_int(nparts)
        _nvtxs = metis_int(nvtxs)
        _ncon = metis_int(ncon)
        objval = metis_int()

        xadj = xadj.astype(metis_int_np)
        adjncy = adjncy.astype(metis_int_np)

        if vwts is not None:
            vwts = vwts.astype(metis_int_np)

        if vsize is not None:
            vsize = vsize.astype(metis_int_np)

        if adjwgt is not None:
            adjwgt = adjwgt.astype(metis_int_np)

        if tpwgts is None:
            tpwgts = np.ones(nparts*ncon, dtype=self.metis_float_np)
            tpwgts /= nparts
        else:
            tpwgts = tpwgts.astype(self.metis_float_np)

        if ubvec is not None:
            ubvec = ubvec.astype(self.metis_float_np)

        if opts is None:
            # Initialize default options
            opts = np.empty(40, dtype=metis_int_np)
            err = self.METIS_SetDefaultOptions(opts.ctypes)

        part = np.empty(nvtxs, dtype=metis_int_np)

        # Run PartGraphKway
        err = self.METIS_PartGraphKway(
            _nvtxs, _ncon, xadj.ctypes, adjncy.ctypes, vwts.ctypes if vwts is not None else None,
            vsize.ctypes if vsize is not None else None,
            adjwgt.ctypes if adjwgt is not None else None, _nparts,
            tpwgts.ctypes, ubvec.ctypes if ubvec is not None else None,
            opts.ctypes, objval, part.ctypes
        )

        if err != 1:
            raise RuntimeError("METIS Error code : {}".format(err))
        else:
            return part
