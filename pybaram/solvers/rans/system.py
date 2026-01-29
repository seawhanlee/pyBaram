# -*- coding: utf-8 -*-
from pybaram.solvers.baseadvecdiff.system import BaseAdvecDiffSystem
from pybaram.solvers.rans import RANSElements, RANSIntInters, RANSBCInters, RANSMPIInters
from pybaram.backends.types import Queue

import numpy as np
import re


class RANSSystem(BaseAdvecDiffSystem):
    name = 'rans'
    _elements_cls = RANSElements
    _intinters_cls = RANSIntInters
    _bcinters_cls = RANSBCInters
    _mpiinters_cls = RANSMPIInters

    def load_solns(self, msh, soln, elemap, cfg, rank):
        # Get initial solution
        if soln:
            for k, ele in elemap.items():
                sol = soln['soln_{}_p{}'.format(k, rank)]
                aux = soln['aux_{}_p{}'.format(k, rank)]

                ele.set_ics_from_sol(sol, aux)
        else:
            # Load btri
            btri = self.load_btri(msh, cfg, rank)

            # Initialize solution
            self.eles.set_ics_from_cfg(btri)

    def load_btri(self, msh, cfg, rank):
        is_loaded = []

        if rank == 0:
            btri = []
            for key in msh:
                m = re.match(r'bcon_([a-z_\d]+)_p([\d]+)$', key)

                if m:
                    # Collect boundary triangles
                    bname = m.group(1)
                    if bname not in is_loaded:
                        is_loaded.append(bname)
                        bcsect = 'soln-bcs-{}'.format(bname)
                        bctype = cfg.get(bcsect, 'type')

                        if bctype in ['adia-wall', 'isotherm-wall']:
                            btri.append(msh['btri_' + bname][:,:self.ndims])
            
            btri = np.vstack(btri)[:,:,:self.ndims]
        else:
            btri = None

        btri = self._comm.bcast(btri, root=0)

        return btri
