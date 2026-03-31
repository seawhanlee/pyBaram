# -*- coding: utf-8 -*-
import numpy as np
import os


def csv_write(fname, header):
    # Write data as CSV file
    outf = open(fname, 'a')

    if os.path.getsize(fname) == 0:
        print(','.join(header), file=outf)

    return outf


class BasePlugin:
    # Abstract class of Plugin
    name = None

    def __init__(self, intg, cfg, suffix):
        # Allocate `soln` array in each element
        for ele in intg.sys.eles:
            if not hasattr(ele, 'soln'):
                ele.soln = np.empty(ele.upts[intg._curr_idx].shape)
