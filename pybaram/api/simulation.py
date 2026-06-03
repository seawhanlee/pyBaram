# -*- coding: utf-8 -*-
from pybaram.backends import get_backend
from pybaram.integrators import get_integrator
from pybaram.utils.mpi import mpi_init

from tqdm import tqdm


def run(mesh, cfg, be='none', comm='none'):
    """
    Fresh run from mesh and configuration files.

    :param mesh: pyBaram NativeReader object
    :type mesh: pyBaram mesh
    :param cfg: pyBaram INIFile object
    :type cfg: config
    :param be: pyBaram backend object
    :type be: Backend
    :param comm: mpi4py comm object
    :type comm: MPI communicator
    """
    # Run common
    _common(mesh, None, cfg, be, comm)


def restart(mesh, soln, cfg, be='none', comm='none'):
    """
    Restarted run from mesh and configuration files.


    :param mesh: pyBaram NativeReader object
    :type mesh: pyBaram mesh
    :param soln: pyBaram NativeReader object
    :type soln: pyBaram solution
    :param cfg: pyBaram INIFile object
    :type cfg: config
    :param be: pyBaram backend object
    :type be: Backend
    :param comm: mpi4py comm object
    :type comm: MPI communicator
    """
    # Check mesh and solution file
    if mesh['mesh_uuid'] != soln['mesh_uuid']:
        raise RuntimeError('Solution is not computed by the mesh')

    # Run common
    _common(mesh, soln, cfg, be, comm)


def _common(msh, soln, cfg, backend, comm):
    if comm == 'none':        
        # Initiate MPI comm world
        comm = mpi_init()

    # Get backend
    if backend == 'none':
        backend = get_backend('cpu', cfg)

    # Get integrator
    integrator = get_integrator(backend, cfg, msh, soln, comm)

    # Add progress bar
    if comm.rank == 0:
        if integrator.mode in ('unsteady', 'unsteady-dts'):
            tend = getattr(integrator, 'tend', integrator.tlist[-1])
            pb = tqdm(
                total=tend, initial=integrator.tcurr,
                unit_scale=True)

            tprev = [integrator.tcurr]

            def callb(intg):
                dt = min(intg.tcurr, tend) - tprev[0]
                tprev[0] = intg.tcurr
                return pb.update(max(dt, 0.0))
        else:
            pb = tqdm(total=integrator.itermax, initial=integrator.iter)

            def callb(intg): return pb.update(1)

        integrator.completed_handler.append(callb)

    integrator.run()
