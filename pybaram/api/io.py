# -*- coding: utf-8 -*-
import os


def import_mesh(inmesh, outmesh, scale=1.0):
    """
    Import genreated mesh to pyBaram.

    :param string inmesh: Original mesh from generator (CGNS, Gmsh)
    :param string outmesh: Converted pyBaram mesh (.pbrm)
    :param float scale: Geometric scale factor 
    """
    import h5py
    from pybaram.readers import get_reader

    # Split ext
    extn = os.path.splitext(inmesh)[1]

    # Get reader
    reader = get_reader(extn, inmesh, scale)

    # Get mesh in the pbm format
    mesh = reader.to_pbm()

    # Save to disk
    with h5py.File(outmesh, 'w') as f:
        for k, v in mesh.items():
            f[k] = v


def partition_mesh(inmesh, outmesh, npart, solns=[]):
    """
    Paritioning pyBarm mesh

    :param string inmesh: path and name of unspliited pyBaram mesh
    :param string outmesh: path and/or name of patitioned mesh
    :param int npart: number of partition
    :param string solution: path and name of patitioned mesh
    """
    from pybaram.partitions import get_partition
    from pybaram.readers.native import NativeReader

    # mesh
    msh = NativeReader(inmesh)

    npart = int(npart)

    if len(solns) > 0:
        solns = [NativeReader(soln) for soln in solns]

    get_partition(msh, outmesh, npart, solns)


def export_soln(meshf, solnf, out, bcs, is_list=False):
    """
    Export solution to visualization file

    :param string mesh: pyBaram mesh file
    :param string soln: pyBaram solution file
    :param string out: exported file for visualization
    :param string bcs: surface to extracted
    :param string is_list: list boundary surfaces
    """
    from pybaram.readers.native import NativeReader
    from pybaram.writers import get_writer

    mesh = NativeReader(meshf)

    if is_list:
        # List boundary surface
        surfs = set(k.split('_')[1] for k in mesh if k.startswith('bcon'))
        for n in surfs:
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
