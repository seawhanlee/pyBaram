import h5py
import numpy as np
import os
import re
import uuid

from collections import defaultdict
from itertools import combinations, product

from pybaram.partitions.metiswrapper import METISWrapper


class METISPartition:
    _wmap = {'tri': 3, 'quad': 4, 'tet': 4, 'pri': 5, 'pyr': 5, 'hex': 6}

    def __init__(self, msh, out, npart, sols, coloring_method='greedy'):
        from pybaram.readers.base import get_mesh_layout

        self.npart = npart
        self.coloring_method = coloring_method

        # Set destination path and file name
        if out.lower().endswith((".pbrm", ".pbrmc")):
            path = os.path.dirname(out) or '.'
            mshname = os.path.basename(out)
        else:
            path=out
            mshname = msh.name
        layout = get_mesh_layout(mshname)

        # Check mesh partition
        msh_part = self._npart(msh)

        # Check solution is none or not
        has_sols = len(sols) > 0
        if has_sols:
            solnames = [os.path.join(path, sol.name) for sol in sols]
        
        # Merge mesh
        if msh_part > 1:
            msh, unions, old_uuid = self._merge_mesh(msh, msh_part)

            if has_sols:
                # Check UUID for sols
                if np.any([sol['mesh_uuid'] != old_uuid for sol in sols]):
                    raise ValueError("Solutions are not matched with the mesh")

                # Collect solution file names and merging solutions
                sols = [self._merge_soln(sol, msh_part, unions, msh['mesh_uuid']) for sol in sols]

        # Partitioning elements and mapping
        mapper = self.partition_mesh(msh, npart)

        # Make new mesh
        newm = defaultdict(list)

        # Update elements, connectivities, vertex and nodes
        # Vertex partitioning has the largest temporary working set.  Do it
        # before element, solution-point, and face arrays accumulate in newm.
        self.partition_vtx(msh, newm, mapper)
        self.partition_elm(msh, newm, mapper)
        self.partition_spt(msh, newm, mapper)
        self.partition_cons(msh, newm, mapper)
        self.partition_bcons(msh, newm, mapper)
        self.copy_nodes(msh, newm)

        # Store rank-local mixed-element ordering or coloring.
        mapper = self.make_rank_layouts(
            newm, npart, layout, mapper,
            coloring_method=coloring_method
        )

        # Assign new UUID
        newm['mesh_uuid'] = np.array(str(uuid.uuid4()), dtype='S')

        # Save new mesh
        mshf = os.path.join(path, os.path.split(mshname)[-1])
        with h5py.File(mshf, 'w') as f:
            for k, v in newm.items():
                f[k] = v

        if has_sols:
            # Partitioning solutions
            for sol, solname in zip(sols, solnames):
                solf = os.path.join(path, os.path.split(solname)[-1])
            
                # Partition solution
                self.partition_soln(sol, mapper, newm['mesh_uuid'], solf)

    def _npart(self, msh):
        # Check number of partion for the given msh
        msh_part = 0
        for k in msh:
            m = re.match(r"con_p(\d+)", k)
            if m:
                msh_part = max(msh_part, int(m.group(1)))
        msh_part += 1

        return msh_part

    def _merge_mesh(self, msh, npart, is_save=False):
        # Mesh dictionary
        newm = defaultdict(list)

        # Find element types
        etypes = sorted({
            k.split('_')[1] for k in msh if k.startswith('elm')
        })
        
        # Infomation for unions {etype : nelm}
        unions = {}
        for etype in etypes:
            nelm = [0]
            for rank in range(npart):
                name = 'elm_{}_p{}'.format(etype, rank)
                if name in msh:
                    n = msh[name].shape[0]

                    # Append elm and spt per rank
                    newm['elm_{}_p0'.format(etype)].append(msh[name])
                    newm['spt_{}_p0'.format(etype)].append(msh['spt_{}_p{}'.format(etype, rank)])
                else:
                    n = 0

                nelm.append(n)               
            
            # Added number for element and rank
            unions[etype] = np.cumsum(nelm)[:-1]

            newm['elm_{}_p0'.format(etype)] = np.vstack(newm['elm_{}_p0'.format(etype)])
            newm['spt_{}_p0'.format(etype)] = np.hstack(newm['spt_{}_p0'.format(etype)])

        # Globalize connectivity
        def globalize_con(lhs, rank, unions, etypes):
            for etype in etypes:
                mask = lhs['f0'] == etype.encode()
                lhs['f1'][mask] += unions[etype][rank]

        # Merge inner connectivity
        lhs, rhs = [], []
        for rank in range(npart):
            l, r = msh['con_p{}'.format(rank)]
            globalize_con(l, rank, unions, etypes)
            globalize_con(r, rank, unions, etypes)

            lhs.append(l)
            rhs.append(r)

        # Merge MPI connectivity
        for (lrank, rrank) in combinations(range(npart), 2):
            lname = 'con_p{}p{}'.format(lrank, rrank)
            if lname in msh:
                rname = 'con_p{}p{}'.format(rrank, lrank)

                l, r = msh[lname], msh[rname]

                globalize_con(l, lrank, unions, etypes)
                globalize_con(r, rrank, unions, etypes)

                lhs.append(l)
                rhs.append(r)

        # Obtain new connectivity
        lhs = np.hstack(lhs)
        rhs = np.hstack(rhs)
        newm['con_p0'] = np.vstack([lhs, rhs])

        # Find the boundary conditions
        bcs = sorted({
            match.group(1)
            for key in msh
            if (match := re.match(r'^bcon_(.+)_p\d+$', key))
        })

        # Merge BC
        for bc in bcs:
            lhs = []
            for rank in range(npart):
                name = 'bcon_{}_p{}'.format(bc, rank)
                if name in msh:
                    l = msh[name]

                    globalize_con(l, rank, unions, etypes)
                    lhs.append(l)
            
            newm['bcon_{}_p0'.format(bc)] = np.hstack(lhs)

        # Collect vertex id and vtx
        vn_chunks = []
        vtx_chunks = []

        for rank in range(npart):
            # Get partitioned vertex
            vtx = msh['vtx_p{}'.format(rank)]

            # Get vertex id
            vn = np.empty(len(vtx), dtype=int)
            for etype in etypes:
                name = 'elm_{}_p{}'.format(etype, rank)
                mask = vtx['f0'] == etype.encode()

                if name in msh:
                    elm = msh[name]
                    vn[mask] = elm[vtx[mask]['f1'], vtx[mask]['f2']]

            globalize_con(vtx, rank, unions, etypes)

            vn_chunks.append(vn)
            vtx_chunks.append(vtx)
            
        vn_all = np.concatenate(vn_chunks)
        vtx_all = np.concatenate(vtx_chunks)

        # Sort all records by vertex id so equal-vertex records are contiguous
        order = np.argsort(vn_all, kind="mergesort")
        vn_s = vn_all[order]
        vtx_s = vtx_all[order]

        # Build compacted CSR over existing vertices
        # uniq_v: sorted vertex ids, cnt: number of records per vertex
        uniq_v, cnt = np.unique(vn_s, return_counts=True)

        ivtx_p0 = np.empty(uniq_v.size + 1, dtype=np.int64)
        ivtx_p0[0] = 0
        np.cumsum(cnt, out=ivtx_p0[1:])

        newm['vtx_p0'] = vtx_s
        newm['ivtx_p0'] = ivtx_p0

        # Copy nodes
        self.copy_nodes(msh, newm)

        # Assign new UUID
        newm['mesh_uuid'] = np.array(str(uuid.uuid4()), dtype='S')
        
        # Save new mesh
        if is_save:
            with h5py.File('merged.pbrm', 'w') as f:
                for k, v in newm.items():
                    f[k] = v

        return newm, unions, msh['mesh_uuid']
    
    def _merge_soln(self, soln, npart, etypes, mesh_uuid):
        # New solution
        news = {}

        # Default vaules
        news['mesh_uuid'] = mesh_uuid
        news['config'] = soln['config']
        news['stats'] = soln['stats']

        # Check aux or not
        is_aux = np.any([k.startswith('aux') for k in soln])

        for etype in etypes:
            sol = []
            if is_aux:
                aux = []

            for rank in range(npart):
                name = 'soln_{}_p{}'.format(etype, rank)

                if name in soln:
                    sol.append(soln[name])

                    if is_aux:
                        aux.append(soln['aux_{}_p{}'.format(etype, rank)])

            news['soln_{}_p0'.format(etype)] = np.hstack(sol)

            if is_aux:
                news['aux_{}_p0'.format(etype)] = np.hstack(aux)

        return news

    def partition_mesh(self, msh, npart):
         # list of elements type
        etypes = []
        for k in msh:
            m = re.match(r'elm_([^_]+)_p0$', k)
            if m:
                etypes.append(m.group(1))
        etypes = sorted(etypes)

        # number of elements
        nele = {t: msh['elm_{}_p0'.format(t)].shape[0] for t in etypes}

        # Do metis Partition
        epart = self._metis_part(npart, etypes, nele, msh['con_p0'])
        epart = epart.astype(int)

        # Mapper etype : (epart, lidx)
        mapper = {}
        i0, i1 = 0, 0
        for t in etypes:
            n = nele[t]
            
            # Partition info for the specific element type
            i1 += n
            lepart = epart[i0:i1]

            # Local index after partitioning for the specific element type
            leidx = np.empty_like(lepart)
            indices = {}
            if n:
                # Cache per-rank element indices once so later partitioning
                # steps do not rebuild ``lepart == p`` masks repeatedly.
                order = np.argsort(lepart, kind='mergesort')
                spart = lepart[order]
                cuts = np.flatnonzero(np.diff(spart)) + 1
                starts = np.r_[0, cuts]
                ends = np.r_[cuts, spart.size]

                for s, e in zip(starts, ends):
                    p = int(spart[s])
                    idx = order[s:e]
                    indices[p] = idx
                    leidx[idx] = np.arange(e - s)

            # Save the mapper for the specific element type
            mapper[t] = {'rank' : lepart, 'local' : leidx, 'indices' : indices}
            i0 += n

        return mapper
    
    def partition_soln(self, soln, mapper, mesh_uuid, solf):
        # New solution
        news = {}

        # Default vaules
        news['mesh_uuid'] = mesh_uuid
        news['config'] = soln['config']
        news['stats'] = soln['stats']

        # Check aux or not
        is_aux = np.any([k.startswith('aux') for k in soln])
        
        for t, lmap in mapper.items():
            sol = soln['soln_{}_p0'.format(t)]

            if is_aux:
                aux = soln['aux_{}_p0'.format(t)]

            for p, eidx in lmap['indices'].items():
                idx = lmap['local'][eidx]

                # Save elm for each rank
                news['soln_{}_p{}'.format(t, p)] = sol[:, eidx][:, idx]

                if is_aux:
                    news['aux_{}_p{}'.format(t, p)] = aux[:, eidx][:, idx]

        # Save new solutioj
        with h5py.File(solf, 'w') as f:
            for k, v in news.items():
                f[k] = v

    def make_rank_layouts(
        self, newm, npart, layout, mapper=None,
        coloring_method='greedy'
    ):
        from pybaram.readers.base import make_rank_layout

        for rank in range(npart):
            local_mapper = make_rank_layout(
                newm, layout, rank, coloring_method=coloring_method
            )
            if mapper is None:
                continue

            for etype, old_to_new in local_mapper.items():
                mask = mapper[etype]['rank'] == rank
                local = mapper[etype]['local'][mask]
                mapper[etype]['local'][mask] = old_to_new[local]

        return mapper

    def _metis_part(self, npart, etypes, nele, con):
        if npart == 1:
            return np.zeros(sum(nele[t] for t in etypes), dtype=int)

        # Weights
        vwgt = np.concatenate([
            np.full(nele[t], self._wmap[t], dtype=np.int64) for t in etypes
        ])

        # Partitioning with METIS
        ne = sum([nele[t] for t in etypes])
        xadj, adjncy = self._global_ele_graph(etypes, nele, con)

        metis = METISWrapper()
        epart = metis.part_graph(npart, ne, xadj, adjncy, vwts=vwgt)

        return epart

    def _global_ele_graph(self, etypes, nele, con):
        from scipy import sparse

        # Offsets for flattening type-local element indices into global ids.
        offsets = {}
        offset = 0
        for t in etypes:
            offsets[t] = offset
            offset += nele[t]

        ne = offset
        lhs, rhs = con

        idx_dtype = (
            np.int32 if ne <= np.iinfo(np.int32).max else np.int64
        )
        lgidx = np.empty(len(lhs), dtype=idx_dtype)
        rgidx = np.empty(len(rhs), dtype=idx_dtype)

        # Convert each side of con_p0 to global element ids.
        for t in etypes:
            etype = t.encode()

            lmask = lhs['f0'] == etype
            rmask = rhs['f0'] == etype

            lgidx[lmask] = offsets[t] + lhs['f1'][lmask]
            rgidx[rmask] = offsets[t] + rhs['f1'][rmask]

        # Build the symmetric adjacency directly as CSR.  COO-to-CSR groups
        # rows and removes duplicate edges without the int64 scalar keys and
        # np.unique sorting workspace used by the previous implementation.
        valid = lgidx != rgidx
        if np.any(valid):
            nedges = 2*np.count_nonzero(valid)
            nvalid = nedges // 2
            src = np.empty(nedges, dtype=idx_dtype)
            dst = np.empty(nedges, dtype=idx_dtype)
            src[:nvalid] = lgidx[valid]
            src[nvalid:] = rgidx[valid]
            dst[:nvalid] = rgidx[valid]
            dst[nvalid:] = lgidx[valid]

            graph = sparse.coo_matrix(
                (np.ones(nedges, dtype=np.int8), (src, dst)),
                shape=(ne, ne)
            ).tocsr()
            graph.sum_duplicates()
            graph.sort_indices()

            xadj = graph.indptr
            adjncy = graph.indices
        else:
            # No internal element adjacency.
            xadj = np.zeros(ne + 1, dtype=idx_dtype)
            adjncy = np.array([], dtype=idx_dtype)

        return xadj, adjncy

    def _localized_con(self, lhs, mapper):
        cpart = np.empty(len(lhs), dtype=np.int32)

        for t, lmap in mapper.items():
            # Mask elements
            mask = lhs['f0'] == t.encode()

            # Global element index
            gidx = lhs['f1'][mask]

            # Obtain partitions for connectivity
            cpart[mask] = lmap['rank'][gidx]

            # Convert global index to local
            lhs['f1'][mask] = lmap['local'][gidx]

        return lhs, cpart

    def partition_elm(self, msh, newm, mapper):
        for t, lmap in mapper.items():
            elm = msh['elm_{}_p0'.format(t)]

            for p, eidx in lmap['indices'].items():
                # Save elm for each rank
                newm['elm_{}_p{}'.format(t, p)] = elm[eidx]
    
    def partition_spt(self, msh, newm, mapper):
        for t, lmap in mapper.items():
            spt = msh['spt_{}_p0'.format(t)]

            for p, eidx in lmap['indices'].items():
                # Save elm for each rank
                newm['spt_{}_p{}'.format(t, p)] = spt[:, eidx]

    def partition_cons(self, msh, newm, mapper):
        lhs, rhs = msh['con_p0']

        # Localized connecvity and rank information
        lhs, lpart = self._localized_con(lhs, mapper)
        rhs, rpart = self._localized_con(rhs, mapper)

        # Sort partition info
        nparts = self.npart
        key = lpart * nparts + rpart

        order = np.argsort(key, kind="mergesort")
        key_s = key[order]

        # Grouping index
        cuts = np.flatnonzero(np.diff(key_s)) + 1
        starts = np.r_[0, cuts]
        ends   = np.r_[cuts, key_s.size]

        # Iterate groups: faces for each (l,r) live in order[starts[j]:ends[j]]
        for s, e in zip(starts, ends):
            # indices of faces in this (l,r) group
            mask = order[s:e]          
            k = key_s[s]
            l = int(k // nparts)
            r = int(k %  nparts)

            if l == r:
                # Internal connectivity
                newm['con_p{}'.format(l)] = [lhs[mask], rhs[mask]]
            else:
                # Keep MPI connectivity as NumPy chunks; converting through
                # Python lists is much slower for structured arrays.
                newm['con_p{}p{}'.format(l, r)].append(lhs[mask])
                newm['con_p{}p{}'.format(r, l)].append(rhs[mask])

        # Save as array
        for k in list(newm):
            if k.startswith('con_'):
                if re.match(r'con_p\d+$', k):
                    # Internal connectivity is stored as paired lhs/rhs rows.
                    newm[k] = np.array(newm[k], dtype='S4,i4,i1,i1')
                elif isinstance(newm[k], list):
                    # MPI connectivity is stored as a single structured array.
                    newm[k] = np.concatenate(newm[k]).astype('S4,i4,i1,i1', copy=False)
                else:
                    newm[k] = np.array(newm[k], dtype='S4,i4,i1,i1')

    def partition_bcons(self, msh, newm, mapper):
        # Partitioning bcons
        for k in msh:
            if k.startswith('bcon'):
                bctype = '_'.join(k.split('_')[1:-1])
                lhs = msh[k]

                # Localized bcon
                lhs, lpart = self._localized_con(lhs, mapper)

                for p in np.unique(lpart):
                    mask = lpart == p
                    newm['bcon_{}_p{}'.format(bctype, p)] = lhs[mask]

    def partition_vtx(self, msh, newm, mapper):
        # Read vtx and ivtx for merged mesh
        vtx, ivtx = msh['vtx_p0'], msh['ivtx_p0']

        # Localized the vtx data
        vtx, vpart = self._localized_con(vtx, mapper)

        # The original incidence records are grouped by global vertex.
        nvtx = ivtx.size - 1
        starts = ivtx[:-1]
        present = np.empty((self.npart, nvtx), dtype=bool)
        local_ids = np.empty((self.npart, nvtx), dtype=np.int32)

        for p in range(self.npart):
            # Boolean grouping is linear and, for the small number of MPI
            # ranks, cheaper than sorting every incidence by partition.
            mask = vpart == p
            newm['vtx_p{}'.format(p)] = vtx[mask]

            # Count partition-p incidences directly over the existing ivtx
            # segments.  This avoids the incidence-sized repeated vtx_id.
            cnt = np.add.reduceat(mask, starts).astype(np.int32, copy=False)
            has_any = cnt != 0
            present[p] = has_any

            pcnt = cnt[has_any]
            pivtx = np.empty(len(pcnt) + 1, dtype=np.int32)
            pivtx[0] = 0
            np.cumsum(pcnt, out=pivtx[1:])
            newm['ivtx_p{}'.format(p)] = pivtx

            # Local vertex IDs increase in global-vertex order, exactly as in
            # the compact ivtx array above.
            np.cumsum(has_any, dtype=np.int32, out=local_ids[p])
            local_ids[p] -= 1

        # A global vertex shared by two ranks contributes its rank-local ID
        # to both directional MPI vertex maps.  Pairwise masks replace the
        # incidence-sized encoded keys and global np.unique operation.
        for p1, p2 in combinations(range(self.npart), 2):
            shared = present[p1] & present[p2]
            if not np.any(shared):
                continue

            newm['nvtx_p{}p{}'.format(p1, p2)] = local_ids[p1, shared]
            newm['nvtx_p{}p{}'.format(p2, p1)] = local_ids[p2, shared]

    def copy_nodes(self, msh, newm):
        # Copy nodes
        newm['nodes'] = msh['nodes']

        # Copy btri
        for k in msh:
            if k.startswith('btri'):
                newm[k] = msh[k]
