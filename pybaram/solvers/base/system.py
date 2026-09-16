# -*- coding: utf-8 -*-
import numpy as np
import re

from collections import OrderedDict
from pybaram.solvers.base import BaseElements, BaseIntInters, BaseBCInters, BaseMPIInters, BaseVRInters, BaseVertex
from pybaram.utils.misc import ProxyList, subclass_by_name
from pybaram.backends.types import Queue


class BaseSystem:
    name = 'base'
    _elements_cls = BaseElements
    _intinters_cls = BaseIntInters
    _mpiinters_cls = BaseMPIInters
    _bcinters_cls = BaseBCInters
    _vrinters_cls = BaseVRInters
    _vertex_cls = BaseVertex

    def __init__(
        self, be, cfg, msh, soln, comm, nreg, impl_op,
        rank_layout_req=None, rank_layout_name=None
    ):
        # Backend
        self.be = be
        
        # Save parallel infos
        self._comm = comm
        self.rank = rank = comm.rank

        # Load elements
        self.eles, elemap = self.load_elements(msh, be, cfg, rank)
        self.ndims = next(iter(self.eles)).ndims

        # load interfaces
        self.iint = self.load_int_inters(msh, be, cfg, rank, elemap)

        # Build rank-local numbering only for implicit solvers.
        rank_layout_data = self._load_rank_cell_layout(
            msh, rank, elemap, impl_op, rank_layout_req,
            rank_layout_name
        )

        # load bc and vr
        self.bint, self.vint = self.load_bc_inters(msh, be, cfg, rank, elemap)

        # load mpiint
        self.mpiint = self.load_mpi_inters(msh, be, cfg, rank, elemap)

        # Load vertex
        self.vertex = vertex = self.load_vertex(msh, be, cfg, rank, elemap)

        # Load solns
        self.load_solns(msh, soln, elemap, cfg, rank)

        # Construct kerenls
        self.eles.construct_kernels(vertex, nreg, impl_op)
        self._construct_rank_implicit_storage(impl_op, rank_layout_data)
        self.load_soln_history(soln, elemap, cfg, rank)
        self.iint.construct_kernels(elemap, impl_op)
        self.bint.construct_kernels(elemap, impl_op)

        # Check reconstructed or not
        self._is_recon = (cfg.getint('solver', 'order', 1) > 1)

        if self.mpiint:
            from mpi4py import MPI

            # Construct MPI kernels
            self.mpiint.construct_kernels(elemap, impl_op)

        # Construct Vertex kernels
        self.vertex.construct_kernels(elemap)

        # Construct queue
        self._queue = Queue()

    def load_elements(self, msh, be, cfg, rank):
        elemap = OrderedDict()
        eles = ProxyList()

        for key in sorted(msh):
            m = re.match(r'spt_([a-z]*)_p{}$'.format(rank), key)

            if m:
                etype = m.group(1)
                spt = msh[m.group(0)]

                # Load elements
                ele = self._elements_cls(be, cfg, etype, spt)
                elemap[etype] = ele
                eles.append(ele)

        return eles, elemap

    def _load_rank_cell_layout(
        self, msh, rank, elemap, impl_op, rank_layout_req=None,
        rank_layout_name=None
    ):
        """Load rank-local cell IDs used by implicit solvers."""
        if impl_op not in ('spectral-radius', 'approx-jacobian'):
            # Explicit solvers do not use rank-wide implicit layouts.
            return None

        eles = list(elemap.values())
        order_keys = {
            ele: 'rorder_{}_p{}'.format(etype, rank)
            for etype, ele in elemap.items()
        }
        color_keys = {
            ele: 'rcolor_{}_p{}'.format(etype, rank)
            for etype, ele in elemap.items()
        }
        order_present = {
            ele: key in msh for ele, key in order_keys.items()
        }
        color_present = {
            ele: key in msh for ele, key in color_keys.items()
        }

        if any(order_present.values()) and not all(order_present.values()):
            raise RuntimeError(
                "Rank cell ordering must be provided for every element type"
            )
        if any(color_present.values()) and not all(color_present.values()):
            raise RuntimeError(
                "Rank cell coloring must be provided for every element type"
            )
        if all(order_present.values()) and all(color_present.values()):
            raise RuntimeError(
                "Mesh cannot contain both rank ordering and coloring"
            )

        if all(order_present.values()):
            layout, cell_idx = self._load_rank_order_ids(msh, order_keys)
        else:
            # Colored and legacy meshes keep element types contiguous in the
            # rank cell space; rcolor supplies only color labels.
            layout = 'rank-coloring' if all(color_present.values()) else 'default'
            cell_idx = self._load_default_rank_order(eles)

        if rank_layout_req and layout != rank_layout_req:
            exts = {'rank-order': '.pbrm', 'rank-coloring': '.pbrmc'}
            req = "{} ({})".format(
                rank_layout_req, exts.get(rank_layout_req, 'unknown')
            )
            got = "{} ({})".format(
                layout, exts.get(layout, 'legacy/default')
            )
            raise RuntimeError(
                "{} requires a {} mesh, but this mesh provides {}".format(
                    rank_layout_name or impl_op, req, got
                )
            )

        neles = self._validate_rank_cell_order(eles, cell_idx)

        # Kernel-side rank-local IDs are int32.  Validate before this cast so
        # invalid or overflowing meshes are caught without wraparound.
        cell_idx = {
            ele: np.asarray(cell_idx[ele], dtype=np.int32)
            for ele in eles
        }

        neighbors = self._construct_rank_neighbors(eles, cell_idx)

        # Persistent rank cell layout consumed by implicit integrators.
        self.rank_neles = neles
        self.rank_cell_ids = cell_idx
        self.rank_layout = layout

        # Construction-only data needed to build face layouts below.
        layout_data = {'neighbors': neighbors}

        if layout == 'rank-coloring':
            color_data = self._load_rank_color_data(
                msh, eles, color_keys, cell_idx, neles
            )
            self.rank_ncolors = color_data['ncolors']
            self.rank_ele_color_order = color_data['ele_color_order']
            self.rank_ele_color_offsets = color_data['ele_color_offsets']
            layout_data['color'] = color_data['color']

        return layout_data

    def _load_rank_order_ids(self, msh, order_keys):
        """Load element-local cell to rank-local cell IDs from rorder."""
        cell_idx = {}
        for ele, key in order_keys.items():
            gids = np.asarray(msh[key], dtype=np.int64)
            if gids.shape != (ele.neles,):
                raise ValueError(
                    "{} must contain one entry per cell".format(key)
                )
            cell_idx[ele] = gids

        return 'rank-order', cell_idx

    def _load_default_rank_order(self, eles):
        """Number cells contiguously by element type in rank-local space."""
        cell_idx = {}
        offset = 0
        for ele in eles:
            cell_idx[ele] = offset + np.arange(ele.neles, dtype=np.int64)
            offset += ele.neles

        return cell_idx

    def _validate_rank_cell_order(self, eles, cell_idx):
        """Validate that rank cell IDs form one rank-local permutation."""
        neles = sum(ele.neles for ele in eles)

        if neles:
            packed = np.concatenate([cell_idx[ele] for ele in eles])
            if (
                packed.min() != 0
                or packed.max() != neles - 1
                or np.unique(packed).size != neles
            ):
                raise ValueError(
                    "Rank cell ordering must be a rank-local permutation"
                )
            if neles > np.iinfo(np.int32).max:
                raise OverflowError(
                    "Rank-local cell storage exceeds 32-bit indexing"
                )

        return neles

    def _construct_rank_neighbors(self, eles, cell_idx):
        """Build element-local face neighbors in rank-local cell IDs."""
        # Boundary and MPI faces start as self-neighbors.  Internal faces
        # overwrite those entries with the opposite cell's rank-local ID.
        neighbors = {
            ele: np.tile(cell_idx[ele], ele.nface).reshape(
                ele.nface, ele.neles
            )
            for ele in eles
        }
        lt, le, lf = self.iint.rawlidx
        rt, re, rf = self.iint.rawridx

        for idx in range(self.iint.nfpts):
            lele = eles[lt[idx]]
            rele = eles[rt[idx]]
            neighbors[lele][lf[idx], le[idx]] = cell_idx[rele][re[idx]]
            neighbors[rele][rf[idx], re[idx]] = cell_idx[lele][le[idx]]

        return neighbors

    def _load_rank_color_data(self, msh, eles, color_keys, cell_idx, neles):
        """Load rcolor data and derive per-element color loop ranges."""
        color = np.empty(neles, dtype=np.int32)
        ele_color_order = {}
        ele_color_offsets = {}
        ele_colors = {}

        for ele, key in color_keys.items():
            values = np.asarray(msh[key], dtype=np.int32)
            if values.shape != (ele.neles,):
                raise ValueError(
                    "{} must contain one entry per cell".format(key)
                )
            if np.any(values < 1):
                raise ValueError(
                    "{} must use positive color numbers".format(key)
                )

            # Rank-wide color labels validate inter-element dependencies and
            # later split colored LU-SGS neighbors into lower/upper sets.
            color[cell_idx[ele]] = values

            # Keep element-local labels to count cells in each rank color.
            ele_colors[ele] = values

            # Element-local cell IDs sorted by color; ele_color_offsets
            # slices this array.
            ele_color_order[ele] = self.be.convert_array(
                np.argsort(values, kind='stable').astype(np.int32)
            )

        lt, le, _ = self.iint.rawlidx
        rt, re, _ = self.iint.rawridx
        # Adjacent cells must be separated by color barriers.
        for idx in range(self.iint.nfpts):
            lele = eles[lt[idx]]
            rele = eles[rt[idx]]
            lridx = cell_idx[lele][le[idx]]
            rridx = cell_idx[rele][re[idx]]
            if color[lridx] == color[rridx]:
                raise ValueError(
                    "Adjacent rank-local cells have the same color"
                )

        ncolors = int(color.max()) if neles else 0

        for ele, values in ele_colors.items():
            counts = np.bincount(values, minlength=ncolors + 1)[1:]

            ele_offsets = np.empty(ncolors + 1, dtype=np.int32)
            ele_offsets[0] = 0
            np.cumsum(counts, out=ele_offsets[1:])

            # Prefix sums give begin/end ranges in ele_color_order[ele].
            ele_color_offsets[ele] = ele_offsets

        return {
            'color': color,
            'ncolors': ncolors,
            'ele_color_order': ele_color_order,
            'ele_color_offsets': ele_color_offsets
        }

    def _construct_rank_implicit_storage(self, impl_op, layout_data):
        """Build rank-wide face lookups and shared implicit face storage.

        Physical faces are numbered once across internal, boundary, and MPI
        interfaces.  The selected rank layout then builds either CSR
        cell-face lookup arrays or dense element-local lookup arrays.
        """
        if impl_op not in ('spectral-radius', 'approx-jacobian'):
            # Explicit solvers do not need rank implicit storage.
            return

        eles = list(self.eles)

        # Construct face areas and, for spectral radii, oriented normals.
        interfaces, face_area, face_normal = (
            self._construct_rank_physical_faces(impl_op)
        )

        if self.rank_layout == 'rank-coloring':
            self._construct_rcolor_face_layout(
                eles, impl_op, layout_data, interfaces,
                face_area, face_normal
            )
        else:
            self._construct_rorder_face_layout(
                eles, impl_op, layout_data, interfaces,
                face_area, face_normal
            )

        if impl_op == 'spectral-radius':
            # One value is shared by the two cells adjacent to an internal
            # face; boundary and MPI faces occupy the same physical-face space.
            self.rank_fspr = self.be.alloc_array((self.rank_nfaces,))
            for inter in interfaces:
                inter.rank_fspr = self.rank_fspr

            if hasattr(eles[0], 'nturbvars'):
                self.rank_tfspr = self.be.alloc_array(
                    (self.rank_nfaces,)
                )
                for inter in interfaces:
                    inter.rank_tfspr = self.rank_tfspr
            return

        nfvars = eles[0].nfvars
        if any(ele.nfvars != nfvars for ele in eles):
            raise ValueError(
                "Rank Jacobian storage requires consistent flow "
                "variable counts across element types"
            )

        # The leading dimension stores the positive and negative split
        # Jacobians.  Face orientation is applied when cell-local data is built.
        self.rank_jmat = self.be.alloc_array(
            (2, nfvars, nfvars, self.rank_nfaces), init=0
        )
        for inter in interfaces:
            inter.rank_jmat = self.rank_jmat

        if hasattr(eles[0], 'nturbvars'):
            nturbvars = eles[0].nturbvars
            if any(ele.nturbvars != nturbvars for ele in eles):
                raise ValueError(
                    "Rank Jacobian storage requires consistent "
                    "turbulence variable counts across element types"
                )

            self.rank_tjmat = self.be.alloc_array(
                (2, nturbvars, nturbvars, self.rank_nfaces), init=0
            )
            for inter in interfaces:
                inter.rank_tjmat = self.rank_tjmat

    def _construct_rank_physical_faces(self, impl_op):
        """Assign rank-local IDs to all physical interface faces."""
        interfaces = [self.iint, *self.bint, *self.mpiint]
        nfaces = sum(inter.nfpts for inter in interfaces)

        if max(nfaces, self.rank_neles) > np.iinfo(np.int32).max:
            raise OverflowError(
                "Rank-local face storage exceeds 32-bit indexing"
            )

        face_area = np.empty(nfaces, dtype=np.float64)
        face_normal = None
        if impl_op == 'spectral-radius':
            # Spectral-radius kernels also need oriented normals.  Shared
            # face IDs and areas are used by both implicit operators.
            face_normal = np.empty((self.ndims, nfaces), dtype=np.float64)

        offset = 0
        for inter in interfaces:
            ids = np.arange(offset, offset + inter.nfpts, dtype=np.int32)
            inter.rank_face_ids = self.be.convert_array(ids)

            face_area[ids] = inter.raw_mag_snorm
            if impl_op == 'spectral-radius':
                face_normal[:, ids] = inter.raw_vec_snorm

            offset += inter.nfpts

        self.rank_nfaces = nfaces

        return interfaces, face_area, face_normal

    def _construct_rorder_face_layout(
        self, eles, impl_op, layout_data, interfaces, face_area, face_normal
    ):
        """Build CSR cell-face lookups for rank-order implicit solvers."""
        rank_neighbors = layout_data['neighbors']
        rank_cell_ids = self.rank_cell_ids
        ncell_faces = sum(ele.nface*ele.neles for ele in eles)
        if ncell_faces > np.iinfo(np.int32).max:
            raise OverflowError(
                "Rank-local face adjacency exceeds 32-bit indexing"
            )

        # CSR rows are rank-local cells; entries are that cell's faces.
        face_indptr = np.empty(self.rank_neles + 1, dtype=np.int32)
        face_indptr[0] = 0
        face_ids = np.full(ncell_faces, -1, dtype=np.int32)
        face_sides = np.zeros(ncell_faces, dtype=np.int8)
        face_neighbors = np.full(ncell_faces, -1, dtype=np.int32)
        rcp_vol = np.empty(self.rank_neles, dtype=np.float64)

        # Each rank cell owns exactly ele.nface entries.
        face_counts = np.empty(self.rank_neles, dtype=np.int32)
        for ele in eles:
            cell_ids = rank_cell_ids[ele]

            for idx in range(ele.neles):
                ridx = cell_ids[idx]
                face_counts[ridx] = ele.nface

        np.cumsum(face_counts, out=face_indptr[1:])

        # Fill per-cell metadata before assigning physical face slots.
        for ele in eles:
            cell_ids = rank_cell_ids[ele]
            neighbors = rank_neighbors[ele]

            for idx in range(ele.neles):
                ridx = cell_ids[idx]
                begin = face_indptr[ridx]
                end = face_indptr[ridx + 1]
                face_neighbors[begin:end] = neighbors[:, idx]
                rcp_vol[ridx] = ele.rcp_vol[idx]

        # Scatter physical face IDs into the owning cell-face entries.
        offset = 0
        for inter in interfaces:
            ids = np.arange(offset, offset + inter.nfpts, dtype=np.int32)

            lt, le, lf = inter.rawlidx
            for idx, face in enumerate(ids):
                # The left side uses the stored face normal orientation.
                lele = eles[lt[idx]]
                ridx = rank_cell_ids[lele][le[idx]]
                pos = face_indptr[ridx] + lf[idx]

                face_ids[pos] = face
                face_sides[pos] = 1

            if hasattr(inter, 'rawridx'):
                rt, re, rf = inter.rawridx
                for idx, face in enumerate(ids):
                    # The right side reuses the same physical face slot with
                    # the opposite orientation.
                    rele = eles[rt[idx]]
                    ridx = rank_cell_ids[rele][re[idx]]
                    pos = face_indptr[ridx] + rf[idx]

                    face_ids[pos] = face
                    face_sides[pos] = -1

            offset += inter.nfpts

        # Every cell-face CSR entry must now point to a physical face slot.
        if np.count_nonzero(face_ids >= 0) != ncell_faces:
            raise RuntimeError(
                "Every cell face must map to a rank-local interface face"
            )

        # Rank-order solvers sweep or assemble over rank cells with this CSR
        # layout.
        self.rank_face_indptr = self.be.convert_array(face_indptr)
        self.rank_face_slots = self.be.convert_array(face_ids)
        self.rank_face_sides = self.be.convert_array(face_sides)
        self.rank_face_neighbors = self.be.convert_array(face_neighbors)

        self.rank_face_area = self.be.convert_array(face_area)
        if impl_op == 'spectral-radius':
            self.rank_face_normal = self.be.convert_array(face_normal)
        self.rank_rcp_vol = self.be.convert_array(rcp_vol)

    def _construct_rcolor_face_layout(
        self, eles, impl_op, layout_data, interfaces, face_area, face_normal
    ):
        """Build element-local face lookups for colored implicit solvers."""
        rank_neighbors = layout_data['neighbors']
        rank_cell_ids = self.rank_cell_ids

        # Colored kernels keep fixed-nface loops per element type.
        ele_face_refs = {}
        ele_face_factors = {}
        if impl_op == 'spectral-radius':
            ele_face_normals = {}
            ele_lower_neighbors = {}
            ele_upper_neighbors = {}
        else:
            ele_neighbors = {}

        for ele in eles:
            # Refs are signed one-based physical face IDs; zero marks an
            # unfilled element face.
            refs = np.zeros((ele.nface, ele.neles), dtype=np.int32)

            # Factors scale each face contribution by face area over cell
            # volume: face_area[face] * ele.rcp_vol[cell].
            factors = np.empty((ele.nface, ele.neles), dtype=np.float64)
            if impl_op == 'spectral-radius':
                normals = np.empty(
                    (self.ndims, ele.nface, ele.neles), dtype=np.float64
                )
            else:
                neighbors = np.asarray(
                    rank_neighbors[ele], dtype=np.int32
                ).copy()

                # Boundary and MPI faces are stored as self-neighbors in the
                # layout; colored sweeps skip them explicitly.
                for idx, ridx in enumerate(rank_cell_ids[ele]):
                    neighbors[neighbors[:, idx] == ridx, idx] = -1

            ele_face_refs[ele] = refs
            ele_face_factors[ele] = factors
            if impl_op == 'spectral-radius':
                ele_face_normals[ele] = normals
            else:
                ele_neighbors[ele] = neighbors

        offset = 0
        for inter in interfaces:
            ids = np.arange(offset, offset + inter.nfpts, dtype=np.int32)

            # A positive ref uses the stored normal orientation; a negative
            # ref reuses the same physical face with the opposite orientation.
            lt, le, lf = inter.rawlidx
            for idx, face in enumerate(ids):
                lele = eles[lt[idx]]
                eidx = le[idx]
                fidx = lf[idx]

                ele_face_refs[lele][fidx, eidx] = face + 1
                ele_face_factors[lele][fidx, eidx] = (
                    face_area[face]*lele.rcp_vol[eidx]
                )
                if impl_op == 'spectral-radius':
                    ele_face_normals[lele][:, fidx, eidx] = (
                        face_normal[:, face]
                    )

            if hasattr(inter, 'rawridx'):
                rt, re, rf = inter.rawridx
                for idx, face in enumerate(ids):
                    rele = eles[rt[idx]]
                    eidx = re[idx]
                    fidx = rf[idx]

                    ele_face_refs[rele][fidx, eidx] = -face - 1
                    ele_face_factors[rele][fidx, eidx] = (
                        face_area[face]*rele.rcp_vol[eidx]
                    )
                    if impl_op == 'spectral-radius':
                        ele_face_normals[rele][:, fidx, eidx] = (
                            -face_normal[:, face]
                        )

            offset += inter.nfpts

        for ele in eles:
            refs = ele_face_refs[ele]
            factors = ele_face_factors[ele]
            if np.count_nonzero(refs) != ele.nface*ele.neles:
                raise RuntimeError(
                    "Every colored element face must map to a rank-local "
                    "interface face"
                )
            ele_face_refs[ele] = self.be.convert_array(refs)
            ele_face_factors[ele] = self.be.convert_array(factors)
            if impl_op == 'spectral-radius':
                ele_face_normals[ele] = self.be.convert_array(
                    ele_face_normals[ele]
                )
            else:
                ele_neighbors[ele] = self.be.convert_array(
                    ele_neighbors[ele]
                )

        ele_cell_ids = {
            ele: self.be.convert_array(
                np.asarray(rank_cell_ids[ele], dtype=np.int32)
            )
            for ele in eles
        }

        # Common colored face layout used by both implicit operators.
        self.rank_ele_face_refs = ele_face_refs
        self.rank_ele_face_factors = ele_face_factors
        self.rank_ele_cell_ids = ele_cell_ids

        if impl_op != 'spectral-radius':
            self.rank_ele_neighbors = ele_neighbors
            return

        # Spectral-radius colored LU-SGS additionally needs oriented normals
        # and color-separated lower/upper dependencies.
        ele_lower_neighbors, ele_upper_neighbors = (
            self._construct_rcolor_lusgs_neighbors(
                eles, rank_neighbors, rank_cell_ids, layout_data['color']
            )
        )

        self.rank_ele_face_normals = ele_face_normals
        self.rank_ele_lower_neighbors = ele_lower_neighbors
        self.rank_ele_upper_neighbors = ele_upper_neighbors

    def _construct_rcolor_lusgs_neighbors(
        self, eles, rank_neighbors, rank_cell_ids, colors
    ):
        """Split colored neighbors into lower and upper LU-SGS dependencies."""
        ele_lower_neighbors = {}
        ele_upper_neighbors = {}

        for ele in eles:
            neighbors = np.asarray(rank_neighbors[ele], dtype=np.int32).copy()
            for idx, ridx in enumerate(rank_cell_ids[ele]):
                neighbors[neighbors[:, idx] == ridx, idx] = -1

            lower = neighbors.copy()
            upper = neighbors.copy()
            ncolor = np.zeros(ele.nface, dtype=np.int32)
            for idx, ridx in enumerate(rank_cell_ids[ele]):
                valid = neighbors[:, idx] >= 0
                ncolor[:] = 0
                ncolor[valid] = colors[neighbors[valid, idx]]
                lower[(~valid) | (ncolor >= colors[ridx]), idx] = -1
                upper[(~valid) | (ncolor <= colors[ridx]), idx] = -1

            ele_lower_neighbors[ele] = self.be.convert_array(lower)
            ele_upper_neighbors[ele] = self.be.convert_array(upper)

        return ele_lower_neighbors, ele_upper_neighbors
    
    def load_solns(self, msh, soln, elemap, cfg, rank):
        # Get initial solution
        if soln:
            for k, ele in elemap.items():
                sol = soln['soln_{}_p{}'.format(k, rank)]

                # Check aux variable is in solution or not
                if 'aux_{}_p{}'.format(k, rank) in soln:
                    aux = soln['aux_{}_p{}'.format(k, rank)]
                else:
                    aux = None

                ele.set_ics_from_sol(sol, aux)
        else:
            self.eles.set_ics_from_cfg()

    def load_soln_history(self, soln, elemap, cfg, rank):
        if not soln:
            return

        mode = cfg.get('solver-time-integrator', 'mode', 'unsteady')
        if mode != 'unsteady-dts':
            return

        names = cfg.get('solver-time-integrator-dts', 'restart-soln-names', '')
        idxs = cfg.get('solver-time-integrator-dts', 'restart-soln-idxs', '')

        if not names or not idxs:
            return

        names = [name.strip() for name in names.split(',')]
        idxs = [int(idx) for idx in idxs.split(',')]

        missing = []

        for name, idx in zip(names, idxs):
            for k, ele in elemap.items():
                key = 'soln_{}_{}_p{}'.format(name, k, rank)
                if key not in soln:
                    missing.append(key)
                    continue

                ele.upts[idx][:] = soln[key]

        if missing and len(missing) != len(names)*len(elemap):
            raise RuntimeError(
                'Restart solution is missing DTS history {}'.format(missing[0])
            )

    def load_int_inters(self, msh, be, cfg, rank, elemap):
        key = 'con_p{0}'.format(rank)
        lhs, rhs = msh[key].astype('U4,i4,i1,i1')
        iint = self._intinters_cls(be, cfg, elemap, lhs, rhs)

        return iint

    def load_mpi_inters(self, msh, be, cfg, rank, elemap):
        mpiint = ProxyList()

        for key in sorted(msh):
            m = re.match(r'con_p{}p(\d+)$'.format(rank), key)

            if m:
                lhs = msh[m.group(0)].astype('U4,i4,i1,i1')
                mpiint.append(self._mpiinters_cls(
                    be, cfg, elemap, lhs, int(m.group(1))))
        return mpiint

    def load_bc_inters(self, msh, be, cfg, rank, elemap):
        bint = ProxyList()
        vint = ProxyList()

        for key in sorted(msh):
            m = re.match(r'bcon_([a-z_\d]+)_p{}$'.format(rank), key)

            if m:
                lhs = msh[m.group(0)].astype('U4,i4,i1,i1')
                name = m.group(1)

                if name.startswith('_virtual_'):
                    # Initiate virtual interfaces
                    vint.append(
                        self._vrinters_cls(be, cfg, elemap, lhs, name[9:])
                    )                        
                    
                else:
                    bcsect = 'soln-bcs-{}'.format(name)
                    bctype = cfg.get(bcsect, 'type')

                    try:
                        # Initiate boundary interfaces
                        bint.append(
                            subclass_by_name(self._bcinters_cls, bctype)
                            (be, cfg, elemap, lhs, m.group(1))
                        )
                    except TypeError as e:
                        print(
                            "Wrong BC: Name: {}, Type: {}, {}".format(name, bctype, e)
                            )

        return bint, vint

    def load_vertex(self, msh, be, cfg, rank, elemap):
        nei_vtx = {}

        for key in sorted(msh):
            m = re.match(r'nvtx_p{}p(\d+)$'.format(rank), key)

            if m:
                p = int(m.group(1))
                nei_vtx.update({p: msh[key]})

        vtx = msh['vtx_p{}'.format(rank)].astype('U4,i4,i1,i1')
        ivtx = msh['ivtx_p{}'.format(rank)]
        vertex = self._vertex_cls(be, cfg, elemap, vtx, ivtx, nei_vtx)

        return vertex
