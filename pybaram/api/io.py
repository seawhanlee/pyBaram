# -*- coding: utf-8 -*-
import os
import re
import uuid

import numpy as np


def _convert_native_mesh(inmesh, layout, scale, coloring_method='greedy'):
    from pybaram.readers.base import convert_rank_layout
    from pybaram.readers.native import NativeReader

    reader = NativeReader(inmesh)
    try:
        mesh = {name: reader[name] for name in reader}
    finally:
        reader.close()

    if scale != 1.0:
        for name in mesh:
            if name == 'nodes' or name.startswith(('spt_', 'btri_')):
                mesh[name] *= scale

        mesh['mesh_uuid'] = np.array(str(uuid.uuid4()), dtype='S')

    return convert_rank_layout(
        mesh, layout, coloring_method=coloring_method
    )


def import_mesh(inmesh, outmesh, scale=1.0, coloring_method='greedy'):
    """
    Import a generated or native mesh into pyBaram.

    :param str inmesh: Input CGNS, Gmsh, or native pyBaram mesh
    :param str outmesh: Output mesh; ``.pbrm`` selects rank ordering and
                       ``.pbrmc`` selects rank coloring
    :param float scale: Geometric scale factor
    :param str coloring_method: Coloring algorithm for ``.pbrmc`` output;
                                ``greedy`` or ``smallest-last``
    """
    import h5py
    from pybaram.readers import get_reader
    from pybaram.readers.base import get_mesh_layout

    # Split ext
    extn = os.path.splitext(inmesh)[1].lower()
    layout = get_mesh_layout(outmesh)

    if extn in ('.pbrm', '.pbrmc'):
        mesh = _convert_native_mesh(
            inmesh, layout, scale, coloring_method=coloring_method
        )
    else:
        # Get reader
        reader = get_reader(extn, inmesh, scale)

        # Get mesh in the pbm format
        mesh = reader.to_pbm(layout, coloring_method=coloring_method)

    # Save to disk
    with h5py.File(outmesh, 'w') as f:
        for k, v in mesh.items():
            f[k] = v


def partition_mesh(
    inmesh, outmesh, npart, solns=[], coloring_method='greedy'
):
    """
    Partition a native pyBaram mesh.

    :param str inmesh: Path to the unpartitioned pyBaram mesh
    :param str outmesh: Output path; ``.pbrm`` selects rank ordering and
                        ``.pbrmc`` selects rank coloring
    :param int npart: Number of partitions
    :param list solns: Solution files to partition with the mesh
    :param str coloring_method: Coloring algorithm for ``.pbrmc`` output;
                                ``greedy`` or ``smallest-last``
    """
    from pybaram.partitions import get_partition
    from pybaram.readers.native import NativeReader

    # mesh
    msh = NativeReader(inmesh)

    npart = int(npart)

    if len(solns) > 0:
        solns = [NativeReader(soln) for soln in solns]

    get_partition(
        msh, outmesh, npart, solns, coloring_method=coloring_method
    )


def export_soln(meshf, solnf, out, bcs, is_list=False):
    """
    Export solution to visualization file

    :param str meshf: pyBaram mesh file
    :param str solnf: pyBaram solution file
    :param str out: Exported visualization file
    :param str bcs: Comma-separated boundary names to export
    :param bool is_list: Print the available boundary names
    """
    from pybaram.readers.native import NativeReader
    from pybaram.writers import get_writer

    mesh = NativeReader(meshf)

    if is_list:
        # List boundary surface
        surfs = {
            match.group(1)
            for key in mesh
            if (match := re.match(r'^bcon_(.+)_p\d+$', key))
        }
        for n in sorted(surfs):
            print(n)
    else:
        soln = NativeReader(solnf)

        # Check solution and mesh are compatible
        if mesh['mesh_uuid'] != soln['mesh_uuid']:
            raise RuntimeError(
                'Solution {} was not computed on mesh {}'.format(solnf, meshf))

        # Get writer
        writer = get_writer(mesh, soln, out, bcs=bcs)

        writer.write()
