# -*- coding: utf-8 -*-
from pybaram.plugins import get_plugin
from pybaram.solvers import get_system
from pybaram.utils.misc import ProxyList


class BaseIntegrator:
    def __init__(self, be, cfg, msh, soln, comm):
        self.be = be
        self.mesh = msh
        self._comm = comm
        
        # Get system of equations
        self.sys = get_system(be, cfg, msh, soln, comm, self.nreg, self.impl_op)
        
        if soln is not None:
            soln.close()

        # Current index for pointing current array
        if not hasattr(self, '_curr_idx'):
            self._curr_idx = 0

        # Check aux array (turbulence variables or others for post processing)
        try:
            self.curr_aux
            self.is_aux = True
        except AttributeError:
            self.is_aux = False

        # Store plugins in the handler
        self.completed_handler = plugins = ProxyList()
        for sect in cfg.sections():
            if sect.startswith('soln-plugin'):
                # Extract plugin name
                name = sect.split('-')[2:]

                # Check plugin has suffix
                if len(name) > 1:
                    name, suffix = name
                else:
                    name, suffix = name[0], None

                # Initiate plugin object and save it to handler
                plugins.append(get_plugin(name, self, cfg, suffix))

    def _make_stage_expr(self, args):
        # Generate formulation of each RK stage.
        return '+'.join(
            '{}*upts[{}][j, idx]'.format(a, i)
            for a, i in zip(args[::2], args[1::2])
        )

    def _compile_stage(self, ele, src):
        gvars = {'nvars': ele.nvars, 'nfvars': getattr(ele, 'nfvars', None)}
        lvars = {}
        exec(src, gvars, lvars)
        return self.be.make_loop(ele.neles, lvars['stage'], src=src)

    @property
    def curr_soln(self):
        # Return current solution array
        return self.soln_at(self._curr_idx)

    def soln_at(self, idx):
        # Return solution array at the given solution bank
        return self.be.get_array(
            self.sys.eles.upts[idx], self.sys.eles.soln
        )

    def restart_soln_idxs(self):
        return []

    @property
    def curr_aux(self):
        # Return current aux variable array
        return self.be.get_array(
            self.sys.eles.aux, self.sys.eles.rawaux
        )

    @property
    def curr_mu(self):
        # Get viscosity variable (mu) vector.
        mu = self.be.get_array(self.sys.eles.mu, self.sys.eles.rawmu)

        if hasattr(self.sys.eles, 'mut'):
            # If turbulent viscosity (mu_t) is defined, return mu + mu_t
            mut = self.be.get_array(self.sys.eles.mut, self.sys.eles.rawmut)
            mu = ProxyList([m1 + m2 for m1, m2 in zip(mu, mut)])
        
        return mu
