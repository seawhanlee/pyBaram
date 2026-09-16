import sys
import os


def mpi_init():
    # Initialize MPI with considering exception.
    # from https://groups.google.com/g/mpi4py/c/me2TFzHmmsQ/m/sSF99LE0t9QJ
    from mpi4py import MPI

    # Communicator
    comm = MPI.COMM_WORLD

    # Modify system exception hook to abot mpi for fatal error.
    sys_excepthook = sys.excepthook

    def mpi_excepthook(type, value, traceback):
        sys_excepthook(type, value, traceback)

        if comm.size > 1:
            sys.stderr.flush()
            comm.Abort(1)
        else:
            MPI.Finalize()            

    sys.excepthook = mpi_excepthook

    return comm


# Code assisted by Codex for obtaining local rank and local size.
# TODO: Verify correctness.

def _parse_positive_int(txt):
    try:
        val = int(txt)
        if val > 0:
            return val
    except (TypeError, ValueError):
        pass
    return None


def _parse_slurm_tasks_per_node(txt):
    """
    Parse Slurm task list forms such as:
    - "4"
    - "4,4"
    - "4(x2)"
    - "2,2,1"
    Return first task count if parseable.
    """
    if not txt:
        return None

    head = txt.split(",")[0].strip()
    if "(x" in head:
        head = head.split("(x", 1)[0].strip()

    return _parse_positive_int(head)


def get_local_rank(comm=None):
    # 1) MPI launcher / scheduler environment variables
    for key in ("OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID"):
        try:
            val = int(os.environ.get(key, ""))
            if val < 0:
                val = None
        except ValueError:
            val = None
        if val is not None:
            return val

    # 2) Fallback to shared-memory communicator
    from mpi4py import MPI

    if comm is None:
        comm = MPI.COMM_WORLD

    local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, 0, MPI.INFO_NULL)
    try:
        return local_comm.Get_rank()
    finally:
        local_comm.Free()


def get_local_size(comm=None):
    # 1) MPI launcher / scheduler environment variables
    for key in ("OMPI_COMM_WORLD_LOCAL_SIZE", "MV2_COMM_WORLD_LOCAL_SIZE"):
        val = _parse_positive_int(os.environ.get(key))
        if val is not None:
            return val

    # Slurm forms can be composite ("4(x2)", "2,2")
    for key in ("SLURM_NTASKS_PER_NODE", "SLURM_STEP_TASKS_PER_NODE"):
        val = _parse_slurm_tasks_per_node(os.environ.get(key))
        if val is not None:
            return val

    # 2) Fallback to shared-memory communicator
    from mpi4py import MPI

    if comm is None:
        comm = MPI.COMM_WORLD

    local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, 0, MPI.INFO_NULL)
    try:
        return local_comm.Get_size()
    finally:
        local_comm.Free()
