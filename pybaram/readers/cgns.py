# -*- coding: utf-8 -*-

from collections import defaultdict

import numpy as np
import re

from pybaram.readers.cgnswrapper import CGNSWrapper
from pybaram.readers.base import BaseReader, ConsAssembler, NodesAssembler


class CGNSZoneReader(object):
    # CGNS element types to petype and node counts
    cgns_map = {
        3: ('line', 2), 5: ('tri', 3), 7: ('quad', 4), 10: ('tet', 4),
        12: ('pyr', 5), 14: ('pri', 6), 17: ('hex', 8),
    }

    def __init__(self, cgns, base, idx):
        self._cgns = cgns
        zone = cgns.zone_read(base, idx)

        # Read nodes
        self.nodepts = self._read_nodepts(zone)

        # Read bc
        bc = self._read_bc(zone)

        # Read elements
        self.elenodes = elenodes = {}
        self.pents = pents = {}

        # Construct elenodes and physical entity
        for idx in range(cgns.nsections(zone)):
            elerng, elenode = self._read_element(zone, idx)

            neles = elerng[1] - elerng[0] + 1
            assigned = np.zeros(neles, dtype=bool)
            for jdx, (bcname, (bcrng, bclist)) in enumerate(bc.items()):
                mask = ((bclist >= elerng[0]) & (bclist <= elerng[1]))
                if np.any(mask):
                    picks = bclist[mask] - elerng[0]
                    if np.any((picks < 0) | (picks >= elerng[1] - elerng[0] + 1)):
                        raise RuntimeError('BC element outside section')

                    if np.any(assigned[picks]):
                        raise RuntimeError(
                            'Overlapping BC element IDs in section'
                        )

                    assigned[picks] = True
                    pent = pents.setdefault(bcname, jdx+1)

                    for k, v in elenode.items():
                        key = k, pent
                        selected = v[picks]
                        if key in elenodes:
                            elenodes[key] = np.concatenate(
                                (elenodes[key], selected), axis=0
                            )
                        else:
                            elenodes[key] = selected

            if not np.any(assigned):
                pent = pents.setdefault('fluid', 0)
                for k, v in elenode.items():
                    key = k, pent
                    if key in elenodes:
                        elenodes[key] = np.concatenate((elenodes[key], v),
                                                       axis=0)
                    else:
                        elenodes[key] = v
            elif not np.all(assigned):
                raise RuntimeError('Unassigned BC elements in section')

    def _read_nodepts(self, zone):
        nnode = zone['size'][0]
        ndim = zone['base']['PhysDim']
        nodepts = np.zeros((3, nnode))

        for i, x in enumerate('XYZ'[:ndim]):
            self._cgns.coord_read(zone, 'Coordinate{}'.format(x), nodepts[i])

        return nodepts

    def _read_bc(self, zone):
        nbc = self._cgns.nbocos(zone)
        bc = {}

        for idx_bc in range(nbc):
            boco = self._cgns.boco_read(zone, idx_bc)
            raw_name = boco['name'].lower()
            name = re.sub(r'[\s-]+', '_', raw_name)
            if name in bc:
                raise RuntimeError(
                    'Duplicate BC name after sanitizing {} to {}'.format(
                        raw_name, name
                    )
                )
            bclist = np.asarray(boco['list'], dtype=self._cgns.int_np)
            if len(np.unique(bclist)) != len(bclist):
                raise RuntimeError('Duplicate element ID in BC {}'.format(name))
            bc[name] = boco['range'], bclist

        return bc

    def _read_element(self, zone, idx):
        s = self._cgns.section_read(zone, idx)

        elerng = s['range']
        conn = np.zeros(s['dim'], dtype=self._cgns.int_np)
        self._cgns.elements_read(s, conn)

        cgns_type = s['etype']
        elenode = {}

        spts = self.cgns_map[cgns_type][1]
        elenode[cgns_type] = conn.reshape(-1, spts)

        return elerng, elenode


class CGNSReader(BaseReader):
    # Supported file types and extensions
    name = 'cgns'
    extn = ['.cgns']

    # CGNS element types to petype and node counts
    _etype_map = CGNSZoneReader.cgns_map

    # Node numbers associated with each element face
    _petype_fnmap = {
        'tri': {'line': [[0, 1], [1, 2], [2, 0]]},
        'quad': {'line': [[0, 1], [1, 2], [2, 3], [3, 0]]},
        'tet': {'tri': [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]},
        'hex': {'quad': [[0, 3, 2, 1], [0, 1, 5, 4], [1, 2, 6, 5],
                         [2, 3, 7, 6], [0, 4, 7, 3], [4, 5, 6, 7]]},
        'pri': {'quad': [[0, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5]],
                'tri': [[0, 2, 1], [3, 4, 5]]},
        'pyr': {'quad': [[0, 3, 2, 1]],
                'tri': [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]}
    }

    def __init__(self, msh, scale):
        # Load and wrap CGNS
        self._cgns = cgns = CGNSWrapper()

        # Read CGNS mesh file
        self._file = file = cgns.open(msh)
        base = cgns.base_read(file, 0)

        # Read zones and stack nodepts, pents and elenodes
        offset = 0
        pent = 0
        pents = {}
        node_chunks = []
        elenode_chunks = defaultdict(list)
        for idx in range(cgns.nzones(base)):
            # read zone
            zone = CGNSZoneReader(cgns, base, idx)
            
            ndims, nn = zone.nodepts.shape

            if idx == 0:
                # Add 1st row to start node number as 1
                node_chunks.append(np.zeros(ndims)[:, None])

            # Stack nodes
            node_chunks.append(zone.nodepts)

            # Collect pents and local mapping in each zone
            pmap = {}
            for k, v in zone.pents.items():
                if k not in pents:
                    pents[k] = pent
                    pent += 1

                pmap[v] = pents[k]

            # Collect elenodes
            for k, v in zone.elenodes.items():
                # Keys as (petype and pent)
                new = k[0], pmap[k[1]]

                # Add offset for global node numbering
                elenode_chunks[new].append(v + offset)

            # Update offset of elenode for next zone
            offset += nn

        # Transpose nodepts
        nodepts = np.hstack(node_chunks).T
        elenodes = {k: np.vstack(v) for k, v in elenode_chunks.items()}

        # Physical entities can be divided up into:
        #  - fluid elements ('the mesh')
        #  - boundary faces
        felespent = pents.pop('fluid')
        bfacespents = {}
        pfacespents = defaultdict(list)

        for name, pent in pents.items():
            if name.startswith('periodic'):
                p = re.match(r'periodic[ _-]([a-z0-9]+)[ _-](l|r)$', name)
                if not p:
                    raise ValueError('Invalid periodic boundary condition')

                pfacespents[p.group(1)].append(pent)
            # Other boundary faces
            else:
                bfacespents[name] = pent

        if any(len(pf) != 2 for pf in pfacespents.values()):
            raise ValueError('Unpaired periodic boundary in mesh')

        # Construct node db
        pents = felespent, bfacespents, pfacespents
        maps = self._etype_map, self._petype_fnmap
        self._cons = ConsAssembler(elenodes, pents, maps, nodepts)
        self._nodes = NodesAssembler(
            nodepts, elenodes, felespent, bfacespents, self._etype_map, scale)

    def __del__(self):
        if hasattr(self, '_file'):
            self._cgns.close(self._file)

    def _to_raw_pbm(self):
        rawm = {}

        rawm.update(self._cons.get_connectivity())
        rawm.update(self._cons.get_vtx_connectivity())
        rawm.update(self._nodes.get_nodes())

        return rawm
