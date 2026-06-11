# -*- coding: utf-8 -*-
from dataclasses import dataclass

import numpy as np


@dataclass
class BSRPattern:
    rowptr: np.ndarray
    colidx: np.ndarray
    diag_slots: np.ndarray
    off_slots: np.ndarray
    ndof: int
    bsize: int


@dataclass
class PETScLinearSystem:
    pattern: BSRPattern
    values: np.ndarray
    A: object
    b: object
    x: object
    ksp: object = None


class PETScElementSolve:
    """Callable PETSc solve for one element group.

    It assembles BSR values, updates the PETSc matrix, solves A x = b,
    and scatters the PETSc solution back to ele.du.
    """
    def __init__(self, system, assemble, scatter, make_ksp, insert_mode):
        self.system = system
        self.assemble = assemble
        self.scatter = scatter
        self.make_ksp = make_ksp
        self.insert_mode = insert_mode

    def __call__(self):
        # Unpack the element group's PETSc matrix, vectors, and BSR layout.
        system = self.system
        pattern = system.pattern
        A, b, x = system.A, system.b, system.x

        # BSR values buffer filled before bulk insertion into PETSc Mat.
        values = system.values

        # Clear reusable storage before assembling this step's linear system.
        values.fill(0)
        x.zeroEntries()

        # Assemble the BSR values and RHS into NumPy/PETSc-owned buffers.
        bv = b.getArray()
        bv.fill(0)
        self.assemble(values, bv)

        # Bulk-insert the assembled BSR values into PETSc's matrix storage.
        A.setValuesBlockedCSR(
            pattern.rowptr, pattern.colidx, values,
            addv=self.insert_mode
        )

        # Finalize PETSc matrix/vector assembly before solving.
        A.assemblyBegin()
        A.assemblyEnd()
        b.assemblyBegin()
        b.assemblyEnd()

        if system.ksp is None:
            system.ksp = self.make_ksp(A)
        else:
            system.ksp.setOperators(A)

        # Solve Krylov subspace linear system
        system.ksp.solve(b, x)

        # Expose x as a read-only NumPy view and copy it into ele.du.
        arr = x.getArray(readonly=True)
        self.scatter(arr)

        # Return iteration number and residual norm.
        return system.ksp.getIterationNumber(), system.ksp.getResidualNorm()


def make_bsr_patterns(eles, nvars):
    patterns = {}

    for ele in eles:
        graph = ele.graph
        indptr = graph['indptr']
        indices = graph['indices']

        # BSR row pointer and block column indices.
        rowptr = [0]
        colidx = []

        # Store BSR block slots for diagonal and face-neighbor contributions.
        diag_slots = np.empty(ele.neles, dtype=np.int64)
        off_slots = np.full((ele.nface, ele.neles), -1, dtype=np.int64)

        for idx in range(ele.neles):
            # Include the diagonal block first, followed by graph neighbors.
            graph_nbrs = indices[indptr[idx]:indptr[idx + 1]]
            nbrs = [idx] + sorted(set(graph_nbrs.tolist()))

            # Map neighbor cell id -> BSR block slot for the current block row.
            row_map = {}

            for nbr in nbrs:
                row_map[nbr] = len(colidx)
                colidx.append(nbr)

            # diag_slots[cell] gives the BSR block slot for A[cell, cell].
            diag_slots[idx] = row_map[idx]

            for jdx in range(ele.nface):
                neib = ele.nei_ele[jdx, idx]

                if neib == idx:
                    continue

                # off_slots[face, cell] gives the BSR block slot for the face neighbor.
                off_slots[jdx, idx] = row_map[neib]

            rowptr.append(len(colidx))

        # Save the PETSc BSR pattern and assembly slot maps for this element group.
        patterns[ele] = BSRPattern(
            rowptr=np.array(rowptr, dtype=np.int32),
            colidx=np.array(colidx, dtype=np.int32),
            diag_slots=diag_slots,
            off_slots=off_slots,
            ndof=ele.neles*nvars,
            bsize=nvars
        )

    return patterns


def make_petsc_systems(patterns, PETSc):
    systems = {}

    for ele, pattern in patterns.items():
        ndof = pattern.ndof
        values = np.zeros(
            len(pattern.colidx)*pattern.bsize*pattern.bsize,
            dtype=np.float64
        )
        systems[ele] = PETScLinearSystem(
            pattern=pattern,

            # Matrix coefficients for the BSR blocks, flattened as values[block_slot*bsize*bsize + row*bsize + col].
            values=values,

            # Preallocated PETSc BAIJ matrix using the block sparsity pattern.
            A=PETSc.Mat().createBAIJ(
                size=(ndof, ndof),
                bsize=pattern.bsize,
                csr=(pattern.rowptr, pattern.colidx),
                comm=PETSc.COMM_SELF
            ),

            # PETSc vectors for the linear system A x = b.
            b=PETSc.Vec().createSeq(ndof, comm=PETSc.COMM_SELF),
            x=PETSc.Vec().createSeq(ndof, comm=PETSc.COMM_SELF)
        )

    return systems
