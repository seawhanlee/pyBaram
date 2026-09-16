from pybaram.partitions.metis import METISPartition


def get_partition(msh, out, npart, sol, coloring_method='greedy'):
    return METISPartition(
        msh, out, npart, sol, coloring_method=coloring_method
    )
