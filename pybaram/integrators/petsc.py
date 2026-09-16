# -*- coding: utf-8 -*-
from dataclasses import dataclass

import numpy as np


@dataclass
class BSRPattern:
    """BSR sparsity and slot maps for PETSc assembly.

    Attributes
    ----------
    rowptr : np.ndarray
        CSR row pointer for rows owned by this rank.
    colidx : np.ndarray
        BSR block columns; rank-local for petsc-rank and global for
        distributed PETSc.
    diag_slots : np.ndarray
        Flat BSR value slots for diagonal cell blocks.
    off_slots : np.ndarray
        Flat BSR value slots for cell-face neighbor blocks.
    local_ndof : int
        Number of degrees of freedom owned by this rank.
    global_ndof : int
        Number of degrees of freedom in the PETSc system.
    bsize : int
        Number of variables per BSR block.
    """
    rowptr: np.ndarray
    colidx: np.ndarray
    diag_slots: np.ndarray
    off_slots: np.ndarray
    local_ndof: int
    global_ndof: int
    bsize: int


class _BasePETScSolve:
    """Reusable PETSc objects for a fixed BSR sparsity pattern."""

    def __init__(self, pattern, assemble, scatter, opts, comm=None):
        self.pattern = pattern
        self.assemble = assemble
        self.scatter = scatter
        self._comm = comm

        # Build reusable PETSc system storage and its KSP solver.
        storage, ksp = self._make_system(pattern, opts)
        self._storage = storage
        self.ksp = ksp

        # Later solves reuse these objects and rebuild only values/RHS.
        self.insert_mode = self._PETSc.InsertMode.INSERT_VALUES

    def _prepare_petsc_assembly(self, pattern):
        try:
            from petsc4py import PETSc
        except ImportError as exc:
            raise RuntimeError(
                "petsc requires petsc4py to be installed"
            ) from exc

        if np.dtype(PETSc.ScalarType) != np.dtype(np.float64):
            raise RuntimeError(
                "petsc requires a real double-precision PETSc build"
            )

        self._PETSc = PETSc
        rowptr = np.asarray(pattern.rowptr, dtype=PETSc.IntType)
        colidx = np.asarray(pattern.colidx, dtype=PETSc.IntType)

        # Backend kernels fill values in PETSc BSR block order.
        values = np.zeros(
            len(pattern.colidx)*pattern.bsize*pattern.bsize,
            dtype=np.float64
        )

        # Return the PETSc handle and reusable CSR/value assembly data.
        return PETSc, rowptr, colidx, values

    def _set_ksp_controls(self, ksp, opts):
        # Apply configured KSP type, tolerances, and iteration limit.
        ksp.setType(opts['type'])
        ksp.setTolerances(
            rtol=opts['rtol'], atol=opts['atol'], max_it=opts['max_it']
        )

    def _configure_asm_ilu(self):
        pass

    def __call__(self):
        A, b, x, values, rowptr, colidx = self._storage
        ksp = self.ksp

        # Reassemble the numerical operator without reallocating PETSc objects.
        values.fill(0)
        x.zeroEntries()

        # Access the PETSc RHS vector as a writable NumPy view.
        bv = b.getArray()
        bv.fill(0)

        # Backend assembly fills matrix values and the PETSc RHS vector view.
        self.assemble(values, bv)

        # Every block in the fixed pattern is overwritten from ``values``.
        A.setValuesBlockedCSR(
            rowptr, colidx, values,
            addv=self.insert_mode
        )

        # Finalize PETSc matrix assembly before solving.
        A.assemblyBegin()
        A.assemblyEnd()

        # Configure ASM+ILU after PETSc has an assembled matrix.
        self._configure_asm_ilu()

        # Solve the linear system into the PETSc solution vector.
        ksp.solve(b, x)

        # Scatter this rank's solution segment into element correction arrays.
        self.scatter(x.getArray(readonly=True))

        # Return iteration count, residual norm, and convergence status.
        reason = int(ksp.getConvergedReason())
        return ksp.getIterationNumber(), ksp.getResidualNorm(), reason


class PETScRankSolve(_BasePETScSolve):
    """Rank-local PETSc solve."""

    def _make_system(self, pattern, opts):
        # Prepare the PETSc handle and matrix assembly buffers.
        PETSc, rowptr, colidx, values = self._prepare_petsc_assembly(pattern)
        ndof = pattern.local_ndof

        # Create rank-local PETSc matrix and vectors.
        A = PETSc.Mat().createBAIJ(
            size=(ndof, ndof),
            bsize=pattern.bsize,
            csr=(rowptr, colidx),
            comm=PETSc.COMM_SELF
        )
        b = PETSc.Vec().createSeq(ndof, comm=PETSc.COMM_SELF)
        x = PETSc.Vec().createSeq(ndof, comm=PETSc.COMM_SELF)

        # Each MPI rank solves its own rank-cell system.
        ksp = PETSc.KSP().create(PETSc.COMM_SELF)
        ksp.setOperators(A)
        self._set_ksp_controls(ksp, opts)

        # Configure the preconditioner.
        pc = ksp.getPC()
        pc.setType(opts['pc'])
        pc.setFactorLevels(opts['pc_levels'])

        # Allow command-line PETSc options to override cfg defaults.
        ksp.setFromOptions()

        return (A, b, x, values, rowptr, colidx), ksp


class PETScGlobalSolve(_BasePETScSolve):
    """Distributed PETSc solve over all MPI ranks."""

    def __init__(self, pattern, assemble, scatter, opts, comm=None):
        super().__init__(pattern, assemble, scatter, opts, comm=comm)

        # Parallel ILU is represented as ASM with local ILU subproblems.
        self._asm_ilu = (
            opts['pc'] == 'ilu'
            and opts['parallel']
            and self.ksp.getPC().getType() == 'asm'
        )
        self._pc_levels = opts['pc_levels']

        # Configure ASM sub-KSPs once, after the first matrix assembly.
        self._asm_configured = False

    def _make_system(self, pattern, opts):
        # Prepare the PETSc handle and matrix assembly buffers.
        PETSc, rowptr, colidx, values = self._prepare_petsc_assembly(pattern)

        # Owned rows are local; block columns are global cell ids.
        comm = (
            PETSc.Comm(self._comm)
            if self._comm is not None else PETSc.COMM_WORLD
        )

        # Create distributed PETSc matrix and vectors.
        A = PETSc.Mat().createBAIJ(
            size=(
                (pattern.local_ndof, pattern.global_ndof),
                (pattern.local_ndof, pattern.global_ndof)
            ),
            bsize=pattern.bsize,
            csr=(rowptr, colidx),
            comm=comm
        )
        b = PETSc.Vec().createMPI(
            (pattern.local_ndof, pattern.global_ndof),
            comm=comm
        )
        x = PETSc.Vec().createMPI(
            (pattern.local_ndof, pattern.global_ndof),
            comm=comm
        )

        # One collective KSP owns the distributed operator.
        ksp = PETSc.KSP().create(comm)
        ksp.setOperators(A)
        self._set_ksp_controls(ksp, opts)

        # Configure the preconditioner.
        pc = ksp.getPC()
        if opts['pc'] == 'ilu' and opts['parallel']:
            # Sub-KSP ILU levels are set after the first matrix assembly.
            pc.setType('asm')
            pc.setASMOverlap(0)
        else:
            pc.setType(opts['pc'])
            if opts['pc'] == 'ilu':
                pc.setFactorLevels(opts['pc_levels'])

        # Allow command-line PETSc options to override cfg defaults.
        ksp.setFromOptions()
        return (A, b, x, values, rowptr, colidx), ksp

    def _configure_asm_ilu(self):
        if self._asm_ilu and not self._asm_configured:
            # ASM sub-KSPs are available only after setup.
            self.ksp.setUp()
            for subksp in self.ksp.getPC().getASMSubKSP():
                subpc = subksp.getPC()
                subpc.setType('ilu')
                subpc.setFactorLevels(self._pc_levels)
            self._asm_configured = True


def make_rank_bsr_pattern(face_indptr, face_neighbors, neles, nvars):
    """Create a rank-cell BSR graph from the cell-face adjacency.

    Each row contains the diagonal and neighbor cell blocks in ascending
    block-column order, as required by PETSc's BAIJ CSR preallocation API.
    """
    iinfo = np.iinfo(np.int32)
    if max(neles, len(face_neighbors)) > iinfo.max:
        raise OverflowError(
            "Rank PETSc BSR pattern exceeds 32-bit internal indexing"
        )

    # Allocate CSR graph arrays and backend slot maps.
    rowptr = [0]
    colidx = []

    # Diagonal slots are indexed by rank cell.
    diag_slots = np.empty(neles, dtype=np.int32)

    # Off-diagonal slots are indexed by cell-face CSR entry.
    off_slots = np.full(len(face_neighbors), -1, dtype=np.int32)

    for ridx in range(neles):
        begin = face_indptr[ridx]
        end = face_indptr[ridx + 1]

        # PETSc BAIJ CSR rows use ascending block-column order.
        nbrs = sorted({
            ridx, *(int(neib) for neib in face_neighbors[begin:end])
        })

        # Append this row's block columns and remember their value slots.
        row_map = {}
        for nbr in nbrs:
            row_map[nbr] = len(colidx)
            colidx.append(nbr)

        # Map each cell and face entry to its BSR value slot.
        diag_slots[ridx] = row_map[ridx]
        for pos in range(begin, end):
            neib = int(face_neighbors[pos])
            if neib != ridx:
                off_slots[pos] = row_map[neib]

        # Close the current CSR row.
        rowptr.append(len(colidx))

    if len(colidx) > iinfo.max:
        raise OverflowError(
            "Rank PETSc BSR pattern exceeds 32-bit internal indexing"
        )

    # Rank-local PETSc has identical local and global vector sizes.
    return BSRPattern(
        rowptr=np.array(rowptr, dtype=np.int32),
        colidx=np.array(colidx, dtype=np.int32),
        diag_slots=diag_slots,
        off_slots=off_slots,
        local_ndof=neles*nvars,
        global_ndof=neles*nvars,
        bsize=nvars
    )


def make_global_bsr_pattern(
    face_indptr, face_neighbors, neles, mpiints, eles, cell_ids, comm,
    nvars
):
    """Create this rank's owned rows of the distributed PETSc BSR matrix.

    Each row stores global diagonal and neighbor blocks in ascending column
    order.
    """
    face_indptr = np.asarray(face_indptr)
    face_neighbors = np.asarray(face_neighbors)

    # Assign this rank a contiguous global cell range.
    counts = comm.allgather(neles)
    cell_offset = sum(counts[:comm.rank])
    nglobal = sum(counts)

    # Convert rank-local neighbor ids to global block-column ids.
    global_neighbors = np.empty(len(face_neighbors), dtype=np.int64)
    for ridx in range(neles):
        begin = face_indptr[ridx]
        end = face_indptr[ridx + 1]
        for pos in range(begin, end):
            global_neighbors[pos] = cell_offset + face_neighbors[pos]

    # Exchange MPI-face owner ids to form off-process BSR columns.
    exchanges = []
    requests = []
    for mpiint in mpiints:
        lt, le, lf = mpiint.rawlidx
        local_gids = np.empty(mpiint.nfpts, dtype=np.int64)
        positions = np.empty(mpiint.nfpts, dtype=np.int64)

        for idx in range(mpiint.nfpts):
            ele = eles[lt[idx]]
            ridx = cell_ids[ele][le[idx]]
            local_gids[idx] = cell_offset + ridx
            positions[idx] = face_indptr[ridx] + lf[idx]

        remote_gids = np.empty_like(local_gids)
        requests.append(comm.Irecv(remote_gids, source=mpiint._dest))
        requests.append(comm.Isend(local_gids, dest=mpiint._dest))
        exchanges.append((positions, remote_gids))

    # Replace MPI self-neighbor placeholders with remote global ids.
    for request in requests:
        request.Wait()
    for positions, remote_gids in exchanges:
        global_neighbors[positions] = remote_gids

    # Allocate CSR graph arrays and backend slot maps.
    rowptr = [0]
    colidx = []

    # Diagonal slots are indexed by rank cell.
    diag_slots = np.empty(neles, dtype=np.int32)

    # Off-diagonal slots are indexed by cell-face CSR entry.
    off_slots = np.full(len(face_neighbors), -1, dtype=np.int32)

    for ridx in range(neles):
        begin = face_indptr[ridx]
        end = face_indptr[ridx + 1]
        row = cell_offset + ridx

        # PETSc BAIJ CSR rows use ascending global block-column order.
        cols = sorted({row, *global_neighbors[begin:end]})
        row_start = len(colidx)
        colidx.extend(cols)

        # Map each owned cell and face entry to its BSR value slot.
        diag_slots[ridx] = row_start + cols.index(row)

        for pos in range(begin, end):
            neighbor = global_neighbors[pos]
            if neighbor != row:
                off_slots[pos] = row_start + cols.index(neighbor)

        # Close the current CSR row.
        rowptr.append(len(colidx))

    # Distributed PETSc owns local rows but uses the global vector size.
    return BSRPattern(
        rowptr=np.asarray(rowptr, dtype=np.int64),
        colidx=np.asarray(colidx, dtype=np.int64),
        diag_slots=diag_slots,
        off_slots=off_slots,
        local_ndof=neles*nvars,
        global_ndof=nglobal*nvars,
        bsize=nvars
    )


def make_rank_petsc_rhs_pack(var0, nvars, factor=1.0):
    def _pack(i_begin, i_end, cell_ids, rhs, dt, diag_slots, a0, av, bv):
        for idx in range(i_begin, i_end):
            ridx = cell_ids[idx]

            # Flattened starts for the rank-cell RHS and diagonal BSR block.
            base = ridx*nvars
            dslot = diag_slots[ridx]*nvars*nvars

            for row in range(nvars):
                # Copy element RHS into the PETSc RHS vector view.
                bv[base + row] = rhs[var0 + row, idx]

                # Add the pseudo-time term to the diagonal matrix entry.
                av[dslot + row*nvars + row] += 1/(dt[idx]*factor) + a0

    return _pack


def make_rank_petsc_turb_pack(be, ele, srcjacobian, factor=1.0):
    nfvars = ele.nfvars
    nvars = ele.nturbvars
    array = be.local()

    def _pack(i_begin, i_end, cell_ids, rhs, upts, dt, dsrc,
              diag_slots, a0, av, bv):
        for idx in range(i_begin, i_end):
            ridx = cell_ids[idx]

            # Flattened starts for the rank-cell RHS and diagonal BSR block.
            base = ridx*nvars
            dslot = diag_slots[ridx]*nvars*nvars
            tmat = array((nvars, nvars), np.float64)

            for row in range(nvars):
                # Copy turbulence RHS into the PETSc RHS vector view.
                bv[base + row] = rhs[nfvars + row, idx]
                for col in range(nvars):
                    tmat[row, col] = 0.0

            # Add the turbulence source Jacobian to the diagonal block.
            srcjacobian(upts[:, idx], tmat, dsrc[:, idx])

            # Add source Jacobian entries to the diagonal BSR block.
            for row in range(nvars):
                for col in range(nvars):
                    av[dslot + row*nvars + col] += tmat[row, col]

            # Add the pseudo-time term to the diagonal matrix entries.
            for row in range(nvars):
                av[dslot + row*nvars + row] += 1/(dt[idx]*factor) + a0

    return _pack


def make_rank_petsc_face_assemble(nvars):
    def _assemble(i_begin, i_end, face_indptr, face_slots, face_sides,
                  face_area, rcp_vol, diag_slots, off_slots, jmat, av):
        bs2 = nvars*nvars

        for ridx in range(i_begin, i_end):
            # Flattened start for this cell's diagonal BSR block.
            dslot = diag_slots[ridx]*bs2

            for pos in range(face_indptr[ridx], face_indptr[ridx + 1]):
                # Locate the physical face and its orientation for this cell.
                slot = face_slots[pos]
                side = face_sides[pos]

                # Face contribution is scaled by area over cell volume.
                fv = face_area[slot]*rcp_vol[ridx]

                # Flattened start for the neighbor BSR block, if present.
                oslot = off_slots[pos]*bs2

                for row in range(nvars):
                    for col in range(nvars):
                        entry = row*nvars + col

                        # Select the diagonal/off-diagonal split for this side.
                        if side == 1:
                            dval = jmat[0, row, col, slot]
                            oval = jmat[1, row, col, slot]
                        else:
                            dval = -jmat[1, row, col, slot]
                            oval = -jmat[0, row, col, slot]

                        # Accumulate diagonal and neighbor face Jacobian terms.
                        av[dslot + entry] += dval*fv
                        if oslot >= 0:
                            av[oslot + entry] += oval*fv

    return _assemble


def make_rank_petsc_scatter(var0, nvars):
    def _scatter(i_begin, i_end, cell_ids, du, sol):
        for idx in range(i_begin, i_end):
            ridx = cell_ids[idx]

            # Flattened start in the rank-cell PETSc solution vector.
            base = ridx*nvars

            for kdx in range(nvars):
                # Copy PETSc solution entries back to element correction SOA.
                du[var0 + kdx, idx] = sol[base + kdx]

    return _scatter


def make_rank_petsc_update(ele):
    nvars = ele.nvars

    def _update(i_begin, i_end, uptsb, dub):
        for idx in range(i_begin, i_end):
            for kdx in range(nvars):
                uptsb[kdx, idx] += dub[kdx, idx]

    return _update
