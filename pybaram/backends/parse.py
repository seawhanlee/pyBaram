# -*- coding: utf-8 -*-
import re
import inspect


_header_pattern = r'def\s(\w+)\(\s*([\w_]+)\s*,\s*([\w_]+)'


def parse_loop(func, src='none', parallel=False):
    # Obtain source
    if src == 'none':
        ftxt = inspect.getsource(func).split('\n')
    else:
        ftxt = src.split('\n')

    # global variables from closure (non nocals and globals)
    closure = inspect.getclosurevars(func)
    gvars = {**closure.nonlocals, **closure.globals}

    # Strip text
    npad = len(ftxt[0]) - len(ftxt[0].lstrip())
    ftxt = [l[npad:] for l in ftxt]

    # Header
    header = ftxt[0]

    # get name and arguments
    m = re.match(_header_pattern, header)
    name = m.group(1)

    # compile regex pattern for main loop
    loop_pattern = r"\s+for\s+[\w_]+\s+in\s+range\({}\s*,\s*{}".format(
        m.group(2), m.group(3)
        )

    # find lines of main loop
    for i, l in enumerate(ftxt):
        if re.match(loop_pattern, l):
            break

    if parallel == 'cpu':
        # replase loop text
        ftxt[i] = re.sub(r"in\s+range\(", 'in nb.prange(', ftxt[i])
    
    # Rewrite header, loop and padded source
    ftxt ='\n'.join(ftxt)
    
    return ftxt, gvars, name