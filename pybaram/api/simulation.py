# -*- coding: utf-8 -*-
from pybaram.backends import get_backend
from pybaram.integrators import get_integrator
from pybaram.api.progress import add_progress_handler
from pybaram.utils.mpi import mpi_init


def run(mesh, cfg, be='none', comm='none', ui='rich', progress_context=None,
        suppress_final_status=False):
    """
    Fresh run from mesh and configuration files.

    :param mesh: pyBaram ``NativeReader`` object
    :type mesh: object
    :param cfg: pyBaram ``INIFile`` object
    :type cfg: object
    :param be: Backend name or backend object
    :type be: str or object
    :param comm: mpi4py communicator
    :type comm: object
    :param ui: progress display mode: 'rich' or 'none'
    :type ui: str
    """
    # Run common
    _common(
        mesh, None, cfg, be, comm, ui, progress_context,
        suppress_final_status
    )


def restart(mesh, soln, cfg, be='none', comm='none', ui='rich',
            progress_context=None, suppress_final_status=False):
    """
    Restarted run from mesh and configuration files.


    :param mesh: pyBaram ``NativeReader`` object
    :type mesh: object
    :param soln: pyBaram solution ``NativeReader`` object
    :type soln: object
    :param cfg: pyBaram ``INIFile`` object
    :type cfg: object
    :param be: Backend name or backend object
    :type be: str or object
    :param comm: mpi4py communicator
    :type comm: object
    :param ui: progress display mode: 'rich' or 'none'
    :type ui: str
    """
    # Check mesh and solution file
    if mesh['mesh_uuid'] != soln['mesh_uuid']:
        raise RuntimeError('Solution is not computed by the mesh')

    # Run common
    _common(
        mesh, soln, cfg, be, comm, ui, progress_context,
        suppress_final_status
    )


def _common(msh, soln, cfg, backend, comm, ui, progress_context,
            suppress_final_status):
    if comm == 'none':        
        # Initiate MPI comm world
        comm = mpi_init()

    # Get backend
    if backend == 'none' or backend is None:
        backend = get_backend('cpu', cfg)
    elif isinstance(backend, str):
        bename = backend.lower()
        if bename in ('cpu', 'cuda'):
            backend = get_backend(bename, cfg, comm=comm)
        else:
            raise ValueError(f'Unsupported backend: {backend}')

    # Get integrator
    integrator = get_integrator(backend, cfg, msh, soln, comm)
    integrator._suppress_final_status = suppress_final_status

    # Add progress display
    progress = add_progress_handler(integrator, comm, ui, progress_context)

    try:
        progress.start()
        integrator.run()
        progress.complete_context(integrator)
    finally:
        progress.stop()
