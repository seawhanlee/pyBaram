# -*- coding: utf-8 -*-

import numpy as np

from pybaram.writers.base import BaseWriter
from pybaram.writers.teciowrapper import TecioWrapper


class TecplotWriter(BaseWriter):
    name = 'plt'
    _is_cstyle = False

    def _raw_write(self):
        # mesh data
        nodes = self._nodes.T.copy()
        nnodes = nodes.shape[1]
        zetype, cons = self._tec_cons()
        ncells = cons.shape[0]

        # Solution data
        sname, sdata, vname, vdata = self._soln
        ndims = self.ndims

        # Variables
        variables = ['X', 'Y', 'Z'][:ndims] + sname

        for vn in vname:
            if vn.startswith('uv'):
                variables += list(vn)
            else:
                variables += ['{}_{}'.format(vn, x) for x in 'xyz'[:max(ndims,2)]]

        try:
            # Binary writing if possible
            self._write_binary(nnodes, ncells, zetype,
                               variables, nodes, cons, sdata, vdata)
        except:
            # Fallback to ascii write
            self._write_ascii(nnodes, ncells, zetype,
                              variables, nodes, cons, sdata, vdata)

    def _tec_cons(self):
        cons = []

        # Convert to Quad and Brick type
        confmap = {
            'line' : (1, lambda e: e),
            'tri': (2, lambda e: [e[0], e[1], e[2], e[2]]),
            'quad': (2, lambda e: e),
            'tet': (3, lambda e: [e[0], e[1], e[2], e[2], e[3], e[3], e[3], e[3]]),
            'pyr': (3, lambda e: [e[0], e[1], e[2], e[3], e[4], e[4], e[4], e[4]]),
            'pri': (3, lambda e: [e[0], e[1], e[2], e[2], e[3], e[4], e[5], e[5]]),
            'hex': (3, lambda e: e)
        }

        for k, v in self._cells.items():
            zetype, conf = confmap[k]
            cons.append(np.array([conf(e) for e in v]))

        return zetype, np.vstack(cons, dtype=np.int32)

    def _write_binary(self, nnodes, ncells, zetype, variables, nodes, cons, sdata, vdata):
        # Construct tecio wrapper
        tecio = TecioWrapper()
        tecio.open(self._outf, variables)

        # Zone type and variable locations
        _zone_etype = {1: 'line', 2: 'quad', 3: 'brick'}
        etype = _zone_etype[zetype]

        valloc = np.zeros(len(variables), dtype=np.int32)
        valloc[:self.ndims] = 1

        # Initialize zone
        tecio.zone(self._outf.split('.')[0],
                   etype, nnodes, ncells, valloc=valloc)

        # Write node and solutions
        tecio.data(nodes[:self.ndims].ravel())
        tecio.data(sdata.ravel())

        tecio.data(vdata.ravel())

        # Write cons
        tecio.node(cons)

        # close
        tecio.close()

    def _write_ascii(self, nnodes, ncells, zetype, variables, nodes, cons, sdata, vdata):
        varlists = " ".join("\"{}\"".format(e) for e in variables)
        centerloc = ','.join(str(e+1) for e in range(self.ndims, len(variables)))

        _zone_type = {1: 'FELINESEG', 2: 'FEQUADRILATERAL', 3: 'FEBRICK'}
        zonet = _zone_type[zetype]

        # Write
        with open(self._outf, 'w') as fp:
            fp.write("VARIABLES = {}\n".format(varlists))
            fp.write("ZONE NODES={}, ELEMENTS={}, DATAPACKING=BLOCK, ZONETYPE={}\nVARLOCATION=([{}]=CELLCENTERED)\n".format(
                nnodes, ncells, zonet, centerloc))

            np.savetxt(fp, nodes[:self.ndims], fmt="%lf", delimiter='\n')

            np.savetxt(fp, sdata, fmt="%E", delimiter='\n')

            np.savetxt(fp, vdata, fmt="%E", delimiter='\n')

            np.savetxt(fp, cons, fmt="%d", delimiter='\n')
