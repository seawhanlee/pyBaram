# -*- coding: utf-8 -*-
from pybaram.backends import get_backend
from pybaram.integrators import get_integrator
from pybaram.api.progress import add_progress_handler
from pybaram.utils.mpi import mpi_init


def run(mesh, cfg, be='none', comm='none', ui='tqdm', progress_context=None):
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
    :param ui: progress display mode: 'tqdm', 'tui', or 'none'
    :type ui: str
    """
    # Run common
    _common(mesh, None, cfg, be, comm, ui, progress_context)


def restart(mesh, soln, cfg, be='none', comm='none', ui='tqdm',
            progress_context=None):
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
    :param ui: progress display mode: 'tqdm', 'tui', or 'none'
    :type ui: str
    """
    # Check mesh and solution file
    if mesh['mesh_uuid'] != soln['mesh_uuid']:
        raise RuntimeError('Solution is not computed by the mesh')

    # Run common
    _common(mesh, soln, cfg, be, comm, ui, progress_context)


def _common(msh, soln, cfg, backend, comm, ui, progress_context):
    if comm == 'none':        
        # Initiate MPI comm world
        comm = mpi_init()

    # Get backend
    if backend == 'none':
        backend = get_backend('cpu', cfg)

    # Get integrator
    integrator = get_integrator(backend, cfg, msh, soln, comm)

    # Add progress display
    progress = add_progress_handler(integrator, comm, ui, progress_context)

    try:
        progress.start()
        integrator.run()
        progress.complete_context(integrator)
    finally:
        progress.stop()
