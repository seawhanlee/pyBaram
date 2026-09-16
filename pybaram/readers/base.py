# -*- coding: utf-8 -*-
# Original code
# https://github.com/PyFR/PyFR/blob/develop/pyfr/readers/base.py
# Modified by jspark
# 
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from itertools import chain
import heapq
import os
import uuid

import numpy as np
import re

from pybaram.utils.np import fuzzysort


class BaseReader(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _to_raw_pbm(self):
        pass

    def to_pbm(self, layout='rank-order', coloring_method='greedy'):
        mesh = self._to_raw_pbm()

        make_rank_layout(
            mesh, layout, coloring_method=coloring_method
        )

        # Add metadata
        mesh['mesh_uuid'] = np.array(str(uuid.uuid4()), dtype='S')

        return mesh


class ConsAssembler(object):
    _con_dtype = np.dtype('S4,i4,i1,i1')

    # Face numberings for each element type
    _petype_fnums = {
        'tri': {'line': [0, 1, 2]},
        'quad': {'line': [0, 1, 2, 3]},
        'tet': {'tri': [0, 1, 2, 3]},
        'hex': {'quad': [0, 1, 2, 3, 4, 5]},
        'pri': {'quad': [0, 1, 2], 'tri': [3, 4]},
        'pyr': {'quad': [0], 'tri': [1, 2, 3, 4]}
    }

    def __init__(self, elenodes, pents, maps, nodepts):
        self._elenodes, self._pents = elenodes, pents
        self._etype_map, self._petype_fnmap = maps
        self._nodepts = nodepts

    def _extract_fluid(self, elenodes, felespent):
        elemap = defaultdict(dict)

        for (etype, pent), eles in elenodes.items():
            petype = self._etype_map[etype][0]
            elemap[pent][petype] = eles

        return elemap.pop(felespent), elemap

    def _extract_faces(self, fpart):
        faces = defaultdict(list)
        for petype, eles in fpart.items():
            for pftype, fnmap in self._petype_fnmap[petype].items():
                fnums = self._petype_fnums[petype][pftype]
                neles, nf = len(eles), len(fnums)

                con = np.empty(neles*nf, dtype=self._con_dtype)
                con['f0'] = petype.encode()
                con['f1'] = np.repeat(np.arange(neles), nf)
                con['f2'] = np.tile(fnums, neles)
                con['f3'] = 0

                nodes = np.sort(eles[:, fnmap], axis=2).reshape(neles*nf, -1)
                faces[pftype].append((con, nodes))

        return faces

    def _pair_fluid_faces(self, faces):
        pairs = defaultdict(list)
        resid = {}

        for pftype, face in faces.items():
            cons, nodes = zip(*face)
            cons = np.concatenate(cons)
            nodes = np.concatenate(nodes)

            if len(nodes) == 0:
                continue

            # Sort faces by their node ids so matching faces are adjacent.
            order = np.lexsort(nodes.T[::-1])
            sorted_nodes = nodes[order]

            same = np.all(sorted_nodes[1:] == sorted_nodes[:-1], axis=1)
            cuts = np.flatnonzero(~same) + 1
            starts = np.r_[0, cuts]
            counts = np.diff(np.r_[starts, len(nodes)])

            paired = counts == 2
            pair_starts = starts[paired]
            resid_starts = starts[~paired]

            # The grouped node rows are no longer needed.  Release them
            # before allocating the paired connectivity result, which is
            # another face-sized array for large meshes.
            del sorted_nodes, same, cuts, starts, counts, paired

            if len(pair_starts):
                # Keep cons in its original order and gather each side of a
                # pair directly.  This avoids sorting/copying every structured
                # connectivity record and halves the temporary integer index
                # storage compared with a flattened two-column pair index.
                pcon = np.empty((len(pair_starts), 2), dtype=self._con_dtype)
                pcon[:, 0] = cons[order[pair_starts]]
                pcon[:, 1] = cons[order[pair_starts + 1]]
                pairs[pftype].append(pcon)

            resid_idx = order[resid_starts]
            for f, n in zip(cons[resid_idx], nodes[resid_idx]):
                resid[tuple(n)] = f

        return pairs, resid

    def _pair_periodic_fluid_faces(self, bparts, resid, pfacespents):
        # paired faces (same as faces)
        pfaces = defaultdict(list)

        # paired bfaces (same as boundary)
        pbfaces = defaultdict(list)

        nodepts = self._nodepts

        for lpent, rpent in pfacespents.values():
            for pftype in bparts[lpent]:
                lfnodes = bparts[lpent][pftype]
                rfnodes = bparts[rpent][pftype]

                lfpts = nodepts[lfnodes]
                rfpts = nodepts[rfnodes]

                lfidx = fuzzysort(lfpts.mean(axis=1).T, range(len(lfnodes)))
                rfidx = fuzzysort(rfpts.mean(axis=1).T, range(len(rfnodes)))

                for lfn, rfn in zip(lfnodes[lfidx], rfnodes[rfidx]):
                    lf = resid.pop(tuple(sorted(lfn)))
                    rf = resid.pop(tuple(sorted(rfn)))

                    pfaces[pftype].append([lf, rf])
                    pbfaces[lpent].append(lf)
                    pbfaces[rpent].append(rf)

        return pfaces, pbfaces

    def _identify_boundary_faces(self, bparts, resid, bfacespents):
        bfaces = defaultdict(list)

        bpents = set(bfacespents.values())

        for pent, fnodes in bparts.items():
            if pent in bpents:
                for fn in chain.from_iterable(fnodes.values()):
                    bfaces[pent].append(resid.pop(tuple(sorted(fn))))

        return bfaces

    def get_connectivity(self):
        felespent, bfacespents, pfacespents = self._pents

        # Extract fluid
        fpart, bparts = self._extract_fluid(self._elenodes, felespent)

        # Extract faces
        faces = self._extract_faces(fpart)

        # Pair faces
        pairs, resid = self._pair_fluid_faces(faces)

        # Periodic faces
        ppairs, pbfaces = self._pair_periodic_fluid_faces(bparts, resid, pfacespents)

        # Identify boundary faces
        bfaces = self._identify_boundary_faces(bparts, resid, bfacespents)

        if any(resid.values()):
            raise ValueError('Unpaired faces in mesh')

        # Connectivity
        con_parts = [p for parts in pairs.values() for p in parts if len(p)]
        con_parts.extend(
            np.asarray(p, dtype=self._con_dtype).reshape(-1, 2)
            for p in ppairs.values() if len(p)
        )
        if con_parts:
            con = np.concatenate(con_parts)
        else:
            con = np.empty((0, 2), dtype=self._con_dtype)

        # Boundary connectivity
        bcon = {}
        for name, pent in bfacespents.items():
            bcon[name] = bfaces[pent]

        # Virtual boundary connectivity
        for name, (lpent, rpent) in pfacespents.items():
            bcon['_virtual_'+name+'_l'] = pbfaces[lpent]
            bcon['_virtual_'+name+'_r'] = pbfaces[rpent]

        # Output
        ret = {'con_p0': con.T}

        for k, v in bcon.items():
            ret['bcon_{0}_p0'.format(k)] = np.array(v, dtype='S4,i4,i1,i1')

        return ret

    def _periodic_vertex_node_map(self, elenodes, pfacespents):
        """Return a node-id map which merges periodic vertex pairs."""
        if not pfacespents:
            return None

        nodepts = self._nodepts
        parent = {}

        # Union-find lets chained periodic pairs collapse to one canonical
        # node id before the vertex records are sorted into groups.
        def root(node):
            parent.setdefault(node, node)
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left, right):
            lroot = root(int(left))
            rroot = root(int(right))
            if lroot != rroot:
                parent[rroot] = lroot

        for etype, pent in elenodes:
            for lpent, rpent in pfacespents.values():
                if lpent != pent:
                    continue

                # Periodic face nodes are matched geometrically by the same
                # fuzzy sort used by the previous vertex-bucket path.
                lnodes = np.unique(elenodes[etype, lpent])
                rnodes = np.unique(elenodes[etype, rpent])

                lpts = nodepts[lnodes]
                rpts = nodepts[rnodes]

                lidx = fuzzysort(lpts.T, range(len(lpts)))
                ridx = fuzzysort(rpts.T, range(len(rpts)))

                for li, ri in zip(lnodes[lidx], rnodes[ridx]):
                    if li != ri:
                        union(li, ri)

        mapped = {
            node: root(node)
            for node in parent
            if root(node) != node
        }
        if not mapped:
            return None

        keys = np.array(sorted(mapped), dtype=np.int64)
        vals = np.array([mapped[key] for key in keys], dtype=np.int64)

        return keys, vals

    def _apply_periodic_vertex_node_map(self, nodes, node_map):
        """Replace periodic node ids by their canonical representatives."""
        if node_map is None:
            return nodes

        keys, vals = node_map

        # keys is sorted, so searchsorted applies the sparse periodic map to
        # the dense element-node vector without building a full node-id array.
        pos = np.searchsorted(keys, nodes)
        valid = pos < len(keys)
        valid[valid] = keys[pos[valid]] == nodes[valid]

        mapped = nodes.copy()
        mapped[valid] = vals[pos[valid]]

        return mapped

    def get_vtx_connectivity(self):
        """Build vertex connectivity without Python node buckets."""
        felespent, pfacespents = self._pents[0], self._pents[-1]
        elenodes = self._elenodes

        nodes_parts = []
        vtx_parts = []
        node_map = self._periodic_vertex_node_map(elenodes, pfacespents)

        for etype, pent in elenodes:
            if pent != felespent:
                continue

            petype, nnode = self._etype_map[etype]
            eles = elenodes[etype, pent]
            neles = len(eles)

            # Build one vertex-connectivity record per element node.  The
            # matching global mesh node id is kept separately in nodes_parts
            # and used only to sort/group these records.
            vtx = np.empty(neles*nnode, dtype=self._con_dtype)
            vtx['f0'] = petype.encode()
            vtx['f1'] = np.repeat(np.arange(neles), nnode)
            vtx['f2'] = np.tile(np.arange(nnode), neles)
            vtx['f3'] = 0

            nodes_parts.append(eles.reshape(-1))
            vtx_parts.append(vtx)

        if not nodes_parts:
            return {
                'vtx_p0': np.empty(0, dtype=self._con_dtype),
                'ivtx_p0': np.zeros(1, dtype=np.int64)
            }

        nodes = np.concatenate(nodes_parts)
        vtx = np.concatenate(vtx_parts)
        del nodes_parts, vtx_parts
        nodes = self._apply_periodic_vertex_node_map(nodes, node_map)

        # Sorting by global node id places all records sharing a vertex next
        # to each other, avoiding a Python dict/list bucket per mesh node.
        order = np.argsort(nodes, kind='mergesort')
        nodes = nodes[order]
        vtx = vtx[order]

        # Group boundaries become ivtx offsets into the already sorted vtx
        # array; the node ids themselves are not needed by the solver.
        cuts = np.flatnonzero(nodes[1:] != nodes[:-1]) + 1
        starts = np.r_[0, cuts]
        ends = np.r_[cuts, len(nodes)]
        ivtx = np.empty(len(starts) + 1, dtype=np.int64)
        ivtx[0] = 0
        np.cumsum(ends - starts, out=ivtx[1:])

        return {'vtx_p0': vtx, 'ivtx_p0': ivtx}


class NodesAssembler(object):
    # Dimensionality of each element type
    _petype_ndim = {'tri': 2, 'quad': 2,
                    'tet': 3, 'hex': 3, 'pri': 3, 'pyr': 3}
    
    def __init__(self, nodepts, elenodes, felespent, bfacespents, etype_map, scale):
        # Scale geometry
        self._nodepts = nodepts*scale

        self._elenodes = elenodes
        self._bfacespents = {v: k for k, v in bfacespents.items()}
        self._felespent = felespent
        self._etype_map = etype_map

    def _fluid_elm(self):
        # Element DB (elm) and points (spt)
        elm = {}
        spt = {}
        for (etype, pent), ele in self._elenodes.items():
            petype = self._etype_map[etype][0]

            if pent == self._felespent:
                elm['elm_{}_p0'.format(petype)] = ele
                spt['spt_{}_p0'.format(petype)] = self._get_spt_ele(petype, ele)

        return elm, spt

    def get_nodes(self):
        # Node points array
        vals = self._nodepts[1:]
        ret = {'nodes': vals}

        # Collect ELments and points
        elm, spt = self._fluid_elm()
        ret.update(elm)
        ret.update(spt)

        # Collect triangulations of boundary
        ret.update(self._extract_btri())

        return ret

    def _get_spt_ele(self, petype, ele):
        ndim = self._petype_ndim[petype]
        nodepts = self._nodepts

        # Get nodes and sort them
        arr = nodepts[ele].swapaxes(0, 1)
        return arr[..., :ndim]

    def _extract_btri(self):
        # Triangulation of boundary surfaces
        btri = defaultdict(list)

        for (etype, pent), ele in self._elenodes.items():
            petype = self._etype_map[etype][0]
            if pent in self._bfacespents:
                bname = self._bfacespents[pent]

                nd_ele = np.array([self._nodepts[v] for v in ele])

                if petype == 'quad':
                    # Pad the center point
                    nd_ele = np.hstack([nd_ele, np.average(nd_ele, axis=1)[:, None]])

                    # Split a quad face as four triangular faces
                    btri[bname].append(nd_ele[:,(0, 1, 4)])
                    btri[bname].append(nd_ele[:,(1, 2, 4)])
                    btri[bname].append(nd_ele[:,(2, 3, 4)])
                    btri[bname].append(nd_ele[:,(3, 0, 4)])
                else:
                    btri[bname].append(nd_ele)

        # Sort the triangulations per surface
        btri = {'btri_{}'.format(k) : np.vstack(v) for k, v in btri.items()}

        return btri
    

def get_mesh_layout(path):
    ext = os.path.splitext(path)[1].lower()
    layouts = {'.pbrm': 'rank-order', '.pbrmc': 'rank-coloring'}

    try:
        return layouts[ext]
    except KeyError:
        raise ValueError(
            "Mesh output extension must be .pbrm or .pbrmc"
        )


def make_rank_layout(
    mshm, layout, rank=0, coloring_method='greedy'
):
    if layout == 'rank-order':
        return make_rank_order(mshm, rank, reorder=True)
    elif layout == 'rank-coloring':
        return make_rank_coloring(
            mshm, rank, coloring_method=coloring_method, reorder=True
        )
    else:
        raise ValueError("Unknown rank-wide mesh layout '{}'".format(layout))


def convert_rank_layout(mshm, layout, coloring_method='greedy'):
    """Replace the rank-wide layout metadata for every mesh partition."""
    for name in list(mshm):
        if name.startswith(('rorder_', 'rcolor_')):
            del mshm[name]

    ranks = set()
    for name in mshm:
        match = re.match(r'elm_[^_]+_p(\d+)$', name)
        if match:
            ranks.add(int(match.group(1)))

    for rank in sorted(ranks):
        make_rank_layout(
            mshm, layout, rank, coloring_method=coloring_method
        )

    return mshm


def _rank_graph(mshm, rank):
    from scipy import sparse

    lhs, rhs = mshm['con_p{}'.format(rank)]

    etypes = sorted(
        k.split('_')[1] for k in mshm
        if k.startswith('elm_') and k.endswith('_p{}'.format(rank))
    )
    nele = {
        etype: len(mshm['elm_{}_p{}'.format(etype, rank)])
        for etype in etypes
    }

    # Contiguous rank-wide ranges for each element type.
    offsets = {}
    offset = 0
    for etype in etypes:
        offsets[etype] = offset
        offset += nele[etype]

    # Convert element-local connectivity records to rank-wide cell IDs.
    neles = offset
    # Scipy's graph routines support 32-bit CSR indices.  Keeping the
    # temporary endpoint arrays at that width substantially reduces both
    # allocation traffic and peak memory for meshes which fit in int32.
    idx_dtype = (
        np.int32 if neles <= np.iinfo(np.int32).max else np.int64
    )
    lridx = np.empty(len(lhs), dtype=idx_dtype)
    rridx = np.empty(len(rhs), dtype=idx_dtype)

    for etype in etypes:
        etype_b = etype.encode()
        lmask = lhs['f0'] == etype_b
        rmask = rhs['f0'] == etype_b

        lridx[lmask] = offsets[etype] + lhs['f1'][lmask]
        rridx[rmask] = offsets[etype] + rhs['f1'][rmask]

    # Build the symmetric adjacency directly as CSR.  COO-to-CSR performs
    # row grouping and duplicate removal without materializing int64 scalar
    # edge keys (src*neles + dst) and their np.unique sorting workspace.
    valid = lridx != rridx
    if np.any(valid):
        nedges = 2*np.count_nonzero(valid)
        src = np.empty(nedges, dtype=idx_dtype)
        dst = np.empty(nedges, dtype=idx_dtype)
        nvalid = nedges // 2
        src[:nvalid] = lridx[valid]
        src[nvalid:] = rridx[valid]
        dst[:nvalid] = rridx[valid]
        dst[nvalid:] = lridx[valid]

        graph = sparse.coo_matrix(
            (np.ones(nedges, dtype=np.int8), (src, dst)),
            shape=(neles, neles)
        ).tocsr()
        graph.sum_duplicates()
        graph.sort_indices()

        indptr = graph.indptr
        indices = graph.indices
    else:
        indptr = np.zeros(neles + 1, dtype=idx_dtype)
        indices = np.array([], dtype=idx_dtype)

    return etypes, nele, offsets, {
        'indptr': indptr, 'indices': indices
    }


def reorder_rank_cells(mshm, rank=0, ordering=None):
    """Physically renumber rank-local element data and connectivity.

    ``ordering`` maps each element type to old element-local indices in the
    desired new order.  When omitted, element-type-local RCM orderings are
    computed from same-type rank-local face adjacencies.
    """
    etypes = sorted(
        k.split('_')[1] for k in mshm
        if k.startswith('elm_') and k.endswith('_p{}'.format(rank))
    )
    if ordering is None:
        lhs, rhs = mshm['con_p{}'.format(rank)]
        nele = {
            etype: len(mshm['elm_{}_p{}'.format(etype, rank)])
            for etype in etypes
        }
        graphs = _etype_rank_graphs(nele, lhs, rhs)
        ordering = {}
        for etype, graph in graphs.items():
            if nele[etype]:
                ordering[etype] = _rcm_by_scipy(graph)
            else:
                ordering[etype] = np.array([], dtype=np.int64)

    old_to_new = {}
    for etype in etypes:
        neles = len(mshm['elm_{}_p{}'.format(etype, rank)])
        idx_dtype = (
            np.int32 if neles <= np.iinfo(np.int32).max else np.int64
        )
        perm = np.asarray(ordering[etype], dtype=idx_dtype)
        if perm.shape != (neles,):
            raise ValueError(
                "Local rank ordering for {} p{} must contain one entry "
                "per cell".format(etype, rank)
            )

        mapper = np.empty(neles, dtype=idx_dtype)
        mapper[perm] = np.arange(neles, dtype=idx_dtype)
        old_to_new[etype] = mapper

        mshm['elm_{}_p{}'.format(etype, rank)] = (
            mshm['elm_{}_p{}'.format(etype, rank)][perm]
        )

        spt_name = 'spt_{}_p{}'.format(etype, rank)
        if spt_name in mshm:
            mshm[spt_name] = mshm[spt_name][:, perm]

    con_name = 'con_p{}'.format(rank)
    _update_con(mshm[con_name][0], old_to_new)
    _update_con(mshm[con_name][1], old_to_new)

    suffix = '_p{}'.format(rank)
    mpi_prefix = 'con_p{}p'.format(rank)
    bcon_names = [
        name for name in mshm
        if name.startswith('bcon_') and name.endswith(suffix)
    ]
    mpi_con_names = [
        name for name in mshm
        if name.startswith(mpi_prefix)
    ]
    for name in bcon_names:
        _update_con(mshm[name], old_to_new)
    for name in mpi_con_names:
        _update_con(mshm[name], old_to_new)

    vtx_name = 'vtx_p{}'.format(rank)
    if vtx_name in mshm:
        _update_con(mshm[vtx_name], old_to_new)

    return old_to_new


def _etype_rank_graphs(nele, lhs, rhs):
    graph = {}

    for etype, neles in nele.items():
        etype_b = etype.encode()
        mask = (lhs['f0'] == etype_b) & (rhs['f0'] == etype_b)

        if np.any(mask):
            # Build same-type directed edges directly from the paired faces.
            # This avoids materializing a large two-row structured array.
            lidx = np.concatenate([lhs['f1'][mask], rhs['f1'][mask]])
            lface = np.concatenate([lhs['f2'][mask], rhs['f2'][mask]])
            ridx = np.concatenate([rhs['f1'][mask], lhs['f1'][mask]])

            # Group by source element so the neighbor list is already in CSR
            # row order; the face index is only a deterministic tie-breaker.
            idx = np.lexsort([lface, lidx])
            lidx = lidx[idx]
            ridx = ridx[idx]

            cuts = np.flatnonzero(lidx[1:] != lidx[:-1]) + 1
            off = np.r_[0, cuts, len(lidx)]
            counts = np.zeros(neles, dtype=np.int64)
            counts[lidx[off[:-1]]] = np.diff(off)

            indptr = np.empty(neles + 1, dtype=np.int64)
            indptr[0] = 0
            np.cumsum(counts, out=indptr[1:])
            indices = ridx.astype(np.int64, copy=False)
        else:
            indptr = np.zeros(neles + 1, dtype=np.int64)
            indices = np.array([], dtype=np.int64)

        graph[etype] = {'indptr': indptr, 'indices': indices}

    return graph


def _update_con(con, mapper):
    cell_ids = con['f1']
    for etype, old_to_new in mapper.items():
        # A boolean mask is one eighth the size of np.nonzero's int64 index
        # array and avoids retaining another large connectivity-sized array
        # while cell data is being physically reordered.
        mask = con['f0'] == etype.encode()
        cell_ids[mask] = old_to_new[cell_ids[mask]]


def make_rank_order(mshm, rank=0, reorder=False):
    """Store rank-wide RCM IDs for each element type.

    The graph is built in the contiguous mixed-element rank space from
    ``_rank_graph``.  The stored ``rorder_*`` arrays map element-local cell
    IDs to the mixed-element rank sweep/order IDs.

    When ``reorder`` is enabled by import/partition, each element type is
    also physically sorted by those rank-wide IDs.  Element arrays cannot
    interleave different types, but this keeps each type's local storage as
    close as possible to the rank-wide RCM sequence.
    """
    etypes, nele, offsets, graph = _rank_graph(mshm, rank)
    neles = len(graph['indptr']) - 1

    if neles:
        permutation = _rcm_by_scipy(graph)

        # RCM returns cells in order; invert it to get cell -> order ID.
        rank_ids = np.argsort(permutation)
        del permutation
    else:
        rank_ids = np.array([], dtype=np.int64)

    # The CSR graph is no longer needed once RCM has completed.  Release it
    # before the largest spt/elm arrays are copied into their reordered form.
    del graph

    order_data = {}
    for etype in etypes:
        begin = offsets[etype]
        end = begin + nele[etype]
        order_data[etype] = rank_ids[begin:end]

    if reorder:
        ordering = {}
        for etype, local_rank_ids in order_data.items():
            # Keep each element type contiguous in storage, but make its local
            # order follow the rank-wide RCM sequence as closely as possible.
            ordering[etype] = np.argsort(local_rank_ids, kind='stable')
            order_data[etype] = local_rank_ids[ordering[etype]]

        # All order_data values now own their sorted data instead of viewing
        # the mixed rank_ids array.
        del rank_ids
        mapper = reorder_rank_cells(mshm, rank, ordering)
    else:
        mapper = None

    # Slice the mixed-element rank IDs back into per-element datasets.
    for etype in etypes:
        mshm['rorder_{}_p{}'.format(etype, rank)] = order_data[etype]

    if reorder:
        return mapper
    else:
        return {
            etype: mshm['rorder_{}_p{}'.format(etype, rank)]
            for etype in etypes
        }


def make_rank_coloring(
    mshm, rank=0, coloring_method='greedy', reorder=False
):
    """Create a rank-wide mixed-element greedy coloring.

    When ``reorder`` is enabled by import/partition, each element type is
    physically grouped by color so colored kernels walk mostly contiguous
    element-local storage inside each color barrier.
    """
    # Build the mixed-element adjacency graph in rank-wide cell IDs.
    etypes, nele, offsets, graph = _rank_graph(mshm, rank)
    indptr, indices = graph['indptr'], graph['indices']
    neles = len(indptr) - 1

    if coloring_method == 'greedy':
        order = _greedy_coloring_order(indptr, indices)
    elif coloring_method == 'smallest-last':
        order = _smallest_last_order(indptr, indices)
    else:
        raise ValueError(
            "Unknown rank coloring method '{}'".format(coloring_method)
        )

    color = _sequential_coloring(indptr, indices, order)

    # Coloring is complete; do not retain the rank graph while copying the
    # largest element and solution-point arrays into color order.
    del graph, indptr, indices, order

    ordering = {}
    color_data = {}
    for etype in etypes:
        begin = offsets[etype]
        end = begin + nele[etype]
        color_data[etype] = color[begin:end]

    if reorder:
        for etype, local_color in color_data.items():
            ordering[etype] = np.argsort(local_color, kind='stable')
            color_data[etype] = local_color[ordering[etype]]

        mapper = reorder_rank_cells(mshm, rank, ordering)
    else:
        mapper = None

    # Slice the rank-wide color vector back into per-element data.
    for etype in etypes:
        mshm['rcolor_{}_p{}'.format(etype, rank)] = color_data[etype]

    if reorder:
        return mapper
    else:
        return {
            etype: mshm['rcolor_{}_p{}'.format(etype, rank)]
            for etype in etypes
        }


def _greedy_coloring_order(indptr, indices):
    """Return high-degree-first greedy coloring order."""
    neles = len(indptr) - 1
    if neles == 0:
        return np.array([], dtype=np.int64)

    degrees = np.diff(indptr)
    return np.lexsort((np.arange(neles), -degrees))


def _sequential_coloring(indptr, indices, order):
    """Color a CSR graph in the given order."""
    neles = len(indptr) - 1
    if neles == 0:
        return np.array([], dtype=np.int32)

    try:
        # graph-tool provides the fast path when it is installed; keep the
        # Python implementation as a dependency-light fallback.
        return _sequential_coloring_by_graph_tool(indptr, indices, order)
    except ImportError:
        return _sequential_coloring_by_python(indptr, indices, order)


def _sequential_coloring_by_graph_tool(indptr, indices, order):
    import graph_tool as gt
    import graph_tool.topology as gtt

    neles = len(indptr) - 1
    idx_dtype = (
        np.int32 if neles <= np.iinfo(np.int32).max else np.int64
    )
    rows = np.repeat(
        np.arange(neles, dtype=idx_dtype), np.diff(indptr)
    )

    # The CSR graph is symmetric; pass each undirected edge to graph-tool
    # once.  Allocate only the retained upper-triangle edge list instead of
    # first materializing a two-column array for every directed edge.
    upper = rows < indices
    edges = np.empty((np.count_nonzero(upper), 2), dtype=idx_dtype)
    edges[:, 0] = rows[upper]
    edges[:, 1] = indices[upper]
    del rows, upper

    graph = gt.Graph(directed=False)
    graph.add_vertex(neles)
    graph.add_edge_list(edges)
    del edges

    order_map = graph.new_vp('int64_t')
    order_map.a = np.asarray(order, dtype=np.int64)

    color = gtt.sequential_vertex_coloring(graph, order_map).a
    # graph-tool colors are zero-based, while pyBaram stores positive labels.
    return color.astype(np.int32, copy=False) + 1


def _sequential_coloring_by_python(indptr, indices, order):
    color = np.zeros(len(indptr) - 1, dtype=np.int32)

    for idx in order:
        used = {
            color[nei]
            for nei in indices[indptr[idx]:indptr[idx + 1]]
            if color[nei] > 0
        }

        current = 1
        while current in used:
            current += 1
        color[idx] = current

    return color


def _smallest_last_order(indptr, indices):
    """Return cells in smallest-last greedy coloring order."""
    neles = len(indptr) - 1
    if neles == 0:
        return np.array([], dtype=np.int64)

    degree = np.diff(indptr).astype(np.int64, copy=True)
    removed = np.zeros(neles, dtype=bool)
    removal = np.empty(neles, dtype=np.int64)
    heap = [(int(degree[idx]), idx) for idx in range(neles)]
    heapq.heapify(heap)

    for pos in range(neles):
        while True:
            deg, idx = heapq.heappop(heap)
            if not removed[idx] and deg == degree[idx]:
                break

        removed[idx] = True
        removal[pos] = idx

        for nei in indices[indptr[idx]:indptr[idx + 1]]:
            if removed[nei]:
                continue

            degree[nei] -= 1
            heapq.heappush(heap, (int(degree[nei]), int(nei)))

    return removal[::-1]


def _rcm_by_scipy(graph):
    # Use Scipy sparse packages
    from scipy import sparse
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    indices, indptr = graph['indices'], graph['indptr']
    nm = len(indptr) - 1

    # Convert graph to sparse matrix
    # RCM only needs the sparsity pattern, so keep the data payload tiny.
    mtx = sparse.csr_matrix(
            (np.ones(indices.size, dtype=np.int8), indices, indptr),
            shape=(nm,nm), copy=False
        )

    return reverse_cuthill_mckee(mtx)
