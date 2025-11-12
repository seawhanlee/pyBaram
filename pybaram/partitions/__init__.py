from pybaram.partitions.metis import METISPartition


def get_partition(msh, out, npart, sol):
    return METISPartition(msh, out, npart, sol)
