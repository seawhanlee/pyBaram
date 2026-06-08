import ctypes
import ctypes.util
import os
import platform


_dll_dir_handles = []


def platform_libnames(name):
    ext = os.path.splitext(os.path.basename(name))[1]
    if ext:
        return [name]

    system = platform.system()
    if system == 'Windows':
        return ['{}.dll'.format(name), 'lib{}.dll'.format(name)]
    elif system == 'Darwin':
        return ['lib{}.dylib'.format(name), '{}.dylib'.format(name)]
    elif system == 'Linux':
        return ['lib{}.so'.format(name), '{}.so'.format(name)]
    else:
        return [ctypes.util.find_library(name) or name]


def platform_dirs():
    # Find environment
    libpaths = os.environ.get('PYBARAM_LIB_PATH', '')

    # Add path for virtualenv
    virtpath = os.environ.get('VIRTUAL_ENV', '')
    if virtpath:
        libpaths = os.pathsep.join([
            libpaths,
            os.path.join(virtpath, 'lib'),
            os.path.join(virtpath, 'Scripts'),
            os.path.join(virtpath, 'Library', 'bin')
        ])

    return [p for p in libpaths.split(os.pathsep) if p]


def _add_dll_directory(path):
    if os.name == 'nt' and hasattr(os, 'add_dll_directory'):
        if os.path.isdir(path):
            _dll_dir_handles.append(os.add_dll_directory(path))


def _load_path(path):
    if os.path.isabs(path):
        _add_dll_directory(os.path.dirname(path))
    return ctypes.CDLL(path)


def load_lib(name):
    # Load library via ctypes
    libnames = platform_libnames(name)
    found = ctypes.util.find_library(name)
    if found:
        libnames.insert(0, found)

    errors = []
    try:
        return _load_path(name)
    except OSError as exc:
        errors.append('{}: {}'.format(name, exc))

    for libname in dict.fromkeys(libnames):
        try:
            return _load_path(libname)
        except OSError as exc:
            errors.append('{}: {}'.format(libname, exc))

        for path in platform_dirs():
            fullpath = os.path.join(path, libname)
            try:
                return _load_path(fullpath)
            except OSError as exc:
                errors.append('{}: {}'.format(fullpath, exc))

    raise OSError('Cannot find {} library. Tried: {}'.format(
        name, '; '.join(errors)
    ))
