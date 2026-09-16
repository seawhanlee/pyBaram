*************
Introduction
*************

Overview
========

pyBaram
-------
pyBaram is an open-source, Python-based software designed to solve compressible flows using the finite volume method on unstructured grids. 'Baram' translates to 'Wind' in Korean. The software supports the simulation of compressible inviscid, laminar, and turbulent flows based on the Reynolds-averaged Navier-Stokes (RANS) models. All the code is written in Python, and hybrid parallel simulations are implemented using high-performance Python packages.

*************
Installation
*************

pyBaram |version| can be obtained from the `repository <https://gitlab.com/aadl_inha/PyBaram>`_.
Currently, ``pyBaram`` supports Linux systems, Windows, and macOS, provided that the required third-party shared libraries are available for the target platform.

Quick start
===========
With `Anaconda <https://www.anaconda.com/>`_ (or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_) Python distribution, you can readily install pyBaram.

1. Make a new environment and activate it::

    user@Computer ~/pyBaram$ conda create -n pybaram python=3.9
    user@Computer ~/pyBaram$ conda activate pybaram

2. Install Python packages::

    user@Computer ~/pyBaram$ conda install numpy scipy numba mpi4py metis
    user@Computer ~/pyBaram$ conda install -c conda-forge h5py cgns

3. Download a release version of ``pyBaram`` from the `release page <https://gitlab.com/aadl_inha/PyBaram/-/releases>`_ and install it::

    user@Computer ~/pyBaram$ pip install pybaram-0.X.Y-py3-none-any.whl


Install from source
===================
You can install pyBaram directly from source using ``setup.py``::

    user@Computer ~/pyBaram$ pip install .

It is recommended to use ``virtualenv`` or ``conda`` to create a separate environment.

Dependencies
------------
pyBaram |version| requires Python 3.9 or newer and the following Python
packages:

1. `numpy` >= 1.10
2. `numba` >= 0.5
3. `scipy` >= 1.6
4. `h5py` >= 2.6
5. `mpi4py` >= 2.0
6. `rich` >= 13.0

Optional Python packages
^^^^^^^^^^^^^^^^^^^^^^^^

1. `petsc4py`
2. `graph-tool`
3. `pykdtree >= 1.3`

The ``petsc4py`` package is required only for the ``petsc`` and ``petsc-rank``
implicit relaxation methods. `PETSc <https://petsc.org/>`_ must use real,
double-precision scalars.
The ``petsc`` method uses a distributed PETSc communicator, while
``petsc-rank`` creates an independent ``PETSc.COMM_SELF`` solver on each MPI
rank.

The ``graph-tool`` package accelerates rank coloring during mesh import and
partitioning for the colored LU-SGS and colored block LU-SGS methods. When it
is unavailable, pyBaram uses its built-in sequential Python implementation.

The ``pykdtree`` package accelerates wall-distance searches for RANS
simulations. When it is unavailable, pyBaram uses its SciPy-based
implementation.

Native libraries
^^^^^^^^^^^^^^^^

The ``CGNS >= 3.4`` library is required to import CGNS meshes, and
``METIS >= 5.1`` is required to partition meshes for parallel computation.

On Windows, ``pyBaram`` requires a METIS dynamic library (``metis.dll`` or
``libmetis.dll``); a static or import ``.lib`` file alone cannot be loaded by
``ctypes``. If the conda package does not provide a METIS DLL, install a
prebuilt METIS DLL or build METIS as a shared library.

The `TecIO <https://www.tecplot.com/products/tecio-library/>`_ 2014 library is
required for binary Tecplot output. Without TecIO, pyBaram writes Tecplot
output in ASCII format.

``pyBaram`` loads CGNS, METIS, and TecIO through ``ctypes``. The library search
path can be extended with the ``PYBARAM_LIB_PATH`` environment variable. Use
``:`` to separate paths on Linux and macOS, and ``;`` on Windows.

For example, on Linux or macOS::

    user@Computer ~/pyBaram$ export PYBARAM_LIB_PATH=/path/to/cgns/lib:/path/to/metis/lib

On Windows::

    C:\> set PYBARAM_LIB_PATH=C:\path\to\cgns\bin;C:\path\to\metis\bin

When running inside a conda environment, ``pyBaram`` also searches common conda
library directories such as ``Library\bin`` on Windows.

CUDA support
------------

The optional CUDA backend requires an NVIDIA CUDA-capable GPU and a CUDA driver
supported by the installed version of Numba.
