# -*- coding: utf-8 -*-
from pybaram.backends.base import Backend
from pybaram.backends.cpu.backend import CPUBackend
from pybaram.backends.cuda.backend import GPUBackend
from pybaram.utils.misc import subclass_by_name


def get_backend(name, *args, **kwargs):
    return subclass_by_name(Backend, name)(*args, **kwargs)
