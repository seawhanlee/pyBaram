# -*- coding: utf-8 -*-

from collections import defaultdict, OrderedDict

import numpy as np
import re

from pybaram.inifile import INIFile
from pybaram.solvers import get_fluid
from pybaram.solvers.base import BaseElements


class BaseWriter(object):
    # Dimensionality of each element type
    _petype_ndim = {'tri': 2, 'quad': 2,
                    'tet': 3, 'hex': 3, 'pri': 3, 'pyr': 3}

    def __init__(self, mesh, soln, outf, bcs=None):
        self._outf = outf        

        # Read solution and config
        self._cfg = cfg = INIFile()

        # Handling string for h5py 2 and 3 are different
        try:
            # h5py 2.X
            self._cfg.fromstr(soln['config'])
        except TypeError:
            # h5py 3.X
            self._cfg.fromstr(soln['config'].decode())

        if bcs is None:
            self._extract_volume(mesh, soln, cfg)
        else:
            surf_names = [k.strip() for k in bcs.split(',')]
            self.extract_surface(mesh, soln, cfg, surf_names)

    def _extract_volume(self, mesh, soln, cfg):
        # Read mesh
        self._nodes = self._get_nodes(mesh)
        self._cells = self._get_cells(mesh)

        # Read solution
        self._soln = self._get_soln(soln, cfg)

    def _get_nodes(self, mesh):
        return mesh['nodes']

    def _get_cells(self, mesh):
        cells = defaultdict(list)
        self._ele_rank = ele_rank = []
        for k in mesh:
            m = re.match(r'elm_(\S+)_p(\d+)', k)
            if m:
                cells[m.group(1)].append(mesh[k])
                ele_rank.append((m.group(1), m.group(2)))

        # Convert C-style array
        if self._is_cstyle:
            off = 1
        else:
            off = 0

        for k in sorted(cells):
            v = cells[k]
            cells[k] = np.concatenate(v) - off

        return OrderedDict(cells)

    def _get_soln(self, soln, cfg):
        sol = defaultdict(list)
        aux = defaultdict(list)
        for (etype, p) in self._ele_rank:
            k = 'soln_{}_p{}'.format(etype, p)
            sol[etype].append(soln[k])

            k = 'aux_{}_p{}'.format(etype, p)
            if k in soln:
                aux[etype].append(soln[k])

        # Load Fluid Elements
        fluid_name = cfg.get('solver', 'system')
        self._elms = elms = get_fluid(fluid_name)

        self.ndims = elms.ndims = self._petype_ndim[etype]

        solns = [np.array(elms.conv_to_prim(np.hstack(sol[k]), cfg))
                 for k in sorted(sol)]
        solns = np.hstack(solns)
        varnames = elms.primevars
        
        if len(aux) > 0:
            auxs = np.hstack([np.hstack(aux[k]) for k in sorted(aux)])
            solns = np.vstack([solns, auxs])
            varnames += elms.auxvars

        # Scalar variables
        sidx = [k not in 'uvw' for k in varnames]
        snames = [k for k in varnames if k not in 'uvw']
        sdata = solns[sidx]

        # Velocity vector
        vidx = [k in 'uvw' for k in varnames]
        vnames = [''.join([k for k in varnames if k in 'uvw'])]
        vdata = solns[vidx]

        return snames, sdata, vnames, vdata

    def extract_surface(self, mesh, soln, cfg, surf_names):
        # Extract boundary connectivities
        bcons = self._extract_bcons(mesh, surf_names)

        # Load Fluid Elements
        fluid_name = cfg.get('solver', 'system')
        fluid = get_fluid(fluid_name)
        
        # Obtain elements and data for each element
        elms = defaultdict(list)
        sdata = defaultdict(list)
        vdata = defaultdict(list)

        # Check Viscous flow or not
        is_viscous = hasattr(fluid, "mu_container")

        for rank, bcon in bcons.items():
            for etype in np.unique(bcon['f0']):
                fluid.ndims = self._petype_ndim[etype]

                # Get element and  associated solution, aux data
                spt = mesh['spt_{}_p{}'.format(etype, rank)]
                sol = soln['soln_{}_p{}'.format(etype, rank)]

                if is_viscous:
                    aux = soln['aux_{}_p{}'.format(etype, rank)]

                # Element object
                ele = BaseElements(None, cfg, etype, spt)

                # Get connectivity for element (-1 for Zero-based numbering)
                elm = mesh['elm_{}_p{}'.format(etype, rank)] - 1

                # Obtain element and face index
                eidx = bcon[bcon['f0'] == etype]['f1']
                fidx = bcon[bcon['f0'] == etype]['f2']

                # Calculation for each face element (quad or tri)
                for f in np.unique(fidx):
                    # Masked index
                    mask = fidx == f
                    ftype, fe = ele.geom._face[f]

                    eidx_f, fidx_f = eidx[mask], fidx[mask]

                    # Obtain normal vector and area of faces
                    vec_fnorm = ele.vec_fnorm[fidx_f, :, eidx_f].T
                    mag_fnorm = ele.mag_fnorm[fidx_f, eidx_f]

                    # Primitive Variables
                    prime = fluid.conv_to_prim(sol[:, eidx_f], cfg)
                    rho, p, uvw = prime[0], prime[1], np.array(prime[2:2+fluid.ndims])

                    # Extract elements
                    elms[ftype].append(elm[eidx_f][:, fe])

                    if is_viscous:
                        # Aux variables
                        auxf = aux[:, eidx_f]

                        # Distance between cell center and face center
                        dx = ele.xf[fidx_f, eidx_f] - ele.xc[eidx_f]
                        dxn = np.einsum('ij,ji->j', vec_fnorm, dx)

                        # Tangential velocity
                        vt = uvw - np.einsum('ij,ij->j', vec_fnorm, uvw)*vec_fnorm
                        trac_vel = vt/dxn

                        # Save pressure, viscosity and tracion velocity
                        sdata[ftype].append(np.vstack([rho, p, *auxf]))
                        vdata[ftype].append(np.vstack([mag_fnorm*vec_fnorm, trac_vel]))
                    else:
                        # Save pressure for inviscid flow
                        sdata[ftype].append(np.vstack([rho, p]))
                        vdata[ftype].append(np.vstack([mag_fnorm*vec_fnorm]))

        self.ndims = fluid.ndims

        # Merge data for face type
        elms = OrderedDict({k: np.vstack(v) for k, v in elms.items()})
        sdata = np.hstack([np.hstack(sdata[k]) for k in sorted(sdata)])
        vdata = np.hstack([np.hstack(vdata[k]) for k in sorted(vdata)])

        snames = ['rho', 'p'] 
        vnames = ['n']

        if is_viscous:
            snames += fluid.auxvars
            vnames += ['wsr']
        
        # Compress nodes (only containing used vtx)
        used_vtx = np.sort(np.unique(np.concatenate([
            np.unique(v) for k, v in elms.items()
            ])))
        
        # mapping
        mapping = {v: i for i, v in enumerate(used_vtx.tolist())}
        mapper = np.vectorize(mapping.get)

        # Convert C-style array
        if self._is_cstyle:
            doff = 0
        else:
            doff = 1

        # Apply mapper to face connectivity
        self._cells = OrderedDict({k: mapper(v) + doff for k, v in elms.items()})

        # Nodes
        self._nodes = mesh['nodes'][used_vtx]

        # Save solutions
        self._soln = snames, sdata, vnames, vdata

    def _extract_bcons(self, mesh, surf_names):
        bcons = defaultdict(list)
        for key in mesh:
            # Find BC data
            m = re.match(r'bcon_([a-z_\d]+)_p(\d+)$', key)

            if m and m.group(1) in surf_names:
                bc = m.group(1)
                rank = eval(m.group(2))

                # Parse BC indexes
                bcons[rank].append(mesh['bcon_{}_p{}'.format(bc, rank)].astype("U4,i4,i1,i1"))

        return {k: np.concatenate(v) for k, v in bcons.items()}
    
    def write(self):
        self._raw_write()
