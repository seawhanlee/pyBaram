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

    def __init__(self, msh, out, npart, sols):
        self.npart = npart

        # Set destination path and file name
        if out.endswith(".pbrm"):
            path = os.path.dirname(out) or '.'
            mshname = os.path.basename(out)
        else:
            path=out
            mshname = msh.name

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
        self.partition_elm(msh, newm, mapper)
        self.partition_spt(msh, newm, mapper)
        self.partition_cons(msh, newm, mapper)
        self.partition_bcons(msh, newm, mapper)
        self.partition_vtx(msh, newm, mapper)
        self.copy_nodes(msh, newm)

        # Re-order mesh
        mapper = self.reoder_rank(newm, npart, mapper)

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
        etypes = set(k.split('_')[1] for k in msh if k.startswith('elm'))
        
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
        bcs = set(k.split('_')[1] for k in msh if k.startswith('bcon'))

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
        etypes = [n.split('_')[1] for n in msh if n.startswith('elm')]

        # number of elements
        nele = {t: msh['elm_{}_p0'.format(t)].shape[0] for t in etypes}

        # Do metis Partition
        epart = self._metis_part(npart, etypes, nele, msh['con_p0'])
        epart = epart.astype(int)

        # Mapper etype : (epart, lidx)
        mapper = {}
        i0, i1 = 0, 0
        for t in etypes:
            # Numbering for elements
            n = nele[t]
            addr = np.arange(n)
            
            # Partition info for the specific element type
            i1 += n
            lepart = epart[i0:i1]

            # Local index after partitioning for the specific element type
            leidx = np.empty_like(lepart)
            for p in np.unique(lepart):
                mask = addr[lepart == p] 
                leidx[mask] = np.arange(len(mask))

            # Save the mapper for the specific element type
            mapper[t] = {'rank' : lepart, 'local' : leidx}
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

            for p in np.unique(lmap['rank']):
                # Mask for the rank
                mask = lmap['rank'] == p
                idx = lmap['local'][mask]

                # Save elm for each rank
                news['soln_{}_p{}'.format(t, p)] = sol[:, mask][:, idx]

                if is_aux:
                    news['aux_{}_p{}'.format(t, p)] = aux[:, mask][:, idx]

        # Save new solutioj
        with h5py.File(solf, 'w') as f:
            for k, v in news.items():
                f[k] = v

    def reoder_rank(self, newm, npart, mapper):
        from pybaram.readers.base import reorder

        for rank in range(npart):
            # Reorder Local mesh
            lmapper = reorder(newm, rank)

            # Update mapper local address
            for etype, mapping in lmapper.items():
                mask = mapper[etype]['rank'] == rank
                idx = mapper[etype]['local'][mask]
                mapper[etype]['local'][mask] = mapping[idx]

        return mapper

    def _metis_part(self, npart, etypes, nele, con):
        # Weights
        vwgt = []
        for t in etypes:
            vwgt += [self._wmap[t]]*nele[t]
        vwgt = np.array(vwgt)

        # Partitioning with METIS
        ne = sum([nele[t] for t in etypes])
        xadj, adjncy = self._global_ele_graph(etypes, nele, con)

        metis = METISWrapper()
        epart = metis.part_graph(npart, ne, xadj, adjncy, vwts=vwgt)

        return epart

    def _global_ele_graph(self, etypes, nele, con):
        # Offsets for flattening type-local element indices into global ids.
        offsets = {}
        offset = 0
        for t in etypes:
            offsets[t] = offset
            offset += nele[t]

        ne = offset
        lhs, rhs = con

        lgidx = np.empty(len(lhs), dtype=np.int64)
        rgidx = np.empty(len(rhs), dtype=np.int64)

        # Convert each side of con_p0 to global element ids.
        for t in etypes:
            etype = t.encode()

            lmask = lhs['f0'] == etype
            rmask = rhs['f0'] == etype

            lgidx[lmask] = offsets[t] + lhs['f1'][lmask]
            rgidx[rmask] = offsets[t] + rhs['f1'][rmask]

        # Build undirected edges.  Encode (src, dst) as src*ne + dst so the
        # duplicate removal uses a faster 1D unique instead of axis=0.
        src = np.concatenate([lgidx, rgidx])
        dst = np.concatenate([rgidx, lgidx])

        mask = src != dst
        src = src[mask]
        dst = dst[mask]

        if len(src):
            keys = np.unique(src*ne + dst)
            src = keys // ne
            dst = keys % ne

            # CSR: neighbors of element i are adjncy[xadj[i]:xadj[i + 1]].
            counts = np.bincount(src, minlength=ne)
            xadj = np.empty(ne + 1, dtype=np.int64)
            xadj[0] = 0
            np.cumsum(counts, out=xadj[1:])

            adjncy = dst.astype(np.int64)
        else:
            # No internal element adjacency.
            xadj = np.zeros(ne + 1, dtype=np.int64)
            adjncy = np.array([], dtype=np.int64)

        return xadj, adjncy

    def _localized_con(self, lhs, mapper):
        cpart = np.empty(len(lhs), dtype=int)

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

            for p in np.unique(lmap['rank']):
                # Mask for the rank
                mask = lmap['rank'] == p

                # Save elm for each rank
                newm['elm_{}_p{}'.format(t, p)] = elm[mask]
    
    def partition_spt(self, msh, newm, mapper):
        for t, lmap in mapper.items():
            spt = msh['spt_{}_p0'.format(t)]

            for p in np.unique(lmap['rank']):
                # Mask for the rank
                mask = lmap['rank'] == p

                # Save elm for each rank
                newm['spt_{}_p{}'.format(t, p)] = spt[:, mask]

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
                # MPI connectivity
                newm['con_p{}p{}'.format(l, r)].extend(lhs[mask].tolist())
                newm['con_p{}p{}'.format(r, l)].extend(rhs[mask].tolist())

        # Save as array
        for k in newm:
            if k.startswith('con_'):
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

        # Make global vertex id
        nvtx = ivtx.size - 1
        deg = np.diff(ivtx)
        vtx_id = np.repeat(np.arange(nvtx), deg)

        # Group per part
        order = np.argsort(vpart, kind='mergesort')
        vpart_sorted = vpart[order]

        cuts = np.flatnonzero(np.diff(vpart_sorted)) + 1
        starts = np.r_[0, cuts]
        ends = np.r_[cuts, vpart_sorted.size]

        for s, e in zip(starts, ends):
             # indices of incidence entries belonging to partition p
            p = int(vpart_sorted[s])
            idx_p = order[s:e]

            # Partitioned vtx
            newm['vtx_p{}'.format(p)] = vtx[idx_p]

            # Count how many entries of partition p touch each global vertex
            cnt = np.bincount(vtx_id[idx_p], minlength=nvtx)
            has_any = cnt > 0

            # global vertices present in p
            gvtx = np.flatnonzero(has_any)

            # Local CSR pointer over compacted vertex list
            newm['ivtx_p{}'.format(p)] = np.cumsum(cnt[gvtx])

        # Unique (global vertex, partition) pairs define each local vertex.
        vpairs = np.unique(vtx_id.astype(np.int64)*self.npart + vpart)
        gvtx = vpairs // self.npart
        vparts = vpairs % self.npart

        # Assign local vertex ids in the same global-vertex order as ivtx_p.
        local = np.empty_like(vparts)
        n_ivtx_p = np.zeros(self.npart, dtype=int)

        cuts = np.flatnonzero(np.diff(gvtx)) + 1
        starts = np.r_[0, cuts]
        ends = np.r_[cuts, gvtx.size]

        for s, e in zip(starts, ends):
            for i in range(s, e):
                p = vparts[i]
                local[i] = n_ivtx_p[p]
                n_ivtx_p[p] += 1

            if e - s > 1:
                for i, j in combinations(range(s, e), 2):
                    p1, p2 = vparts[i], vparts[j]
                    newm['nvtx_p{}p{}'.format(p1, p2)].append(local[i])
                    newm['nvtx_p{}p{}'.format(p2, p1)].append(local[j])

        # Make numpy array
        for k, v in newm.items():
            if k.startswith('vtx'):
                newm[k] = np.array(v, dtype='S4,i4,i1,i1')
            elif k.startswith('ivtx'):
                newm[k] = np.concatenate([[0], v], dtype='i4')
            elif k.startswith('nvtx'):
                newm[k] = np.array(v, dtype='i4')

    def copy_nodes(self, msh, newm):
        # Copy nodes
        newm['nodes'] = msh['nodes']

        # Copy btri
        for k in msh:
            if k.startswith('btri'):
                newm[k] = msh[k]
