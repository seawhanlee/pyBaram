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
Currently, ``pyBaram`` supports Linux systems and Windows Subsystem for Linux (WSL).

Quick start
===========
With `Anaconda <https://www.anaconda.com/>`_ (or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_) Python distribution, you can readily install pyBaram.

1. Make a new environment and activate it::

    user@Computer ~/pyBaram$ conda create -n pybaram
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
pyBaram |version| requires Python 3.9+ and following python packages.

1. `numpy` >= 1.10
2. `numba` >= 0.5
3. `scipy` >= 1.6
4. `h5py` >= 2.6
5. `mpi4py` >= 2.0
6. `tqdm` >= 4.0

The ``scipy`` package is a required dependency and is used for numerical utilities and sparse matrix operations. Mesh reordering using the reverse Cuthill-McKee algorithm is applied by default.

In order to convert the mesh with CGNS format, CGNS library is required.

1. `CGNS` >= 3.4

To partition the mesh for parallel computation, `METIS` library is required.

1. `METIS` >= 5.1

To convert a solution to `Tecplot <https://www.tecplot.com/>`_ binary format, `TecIO <https://www.tecplot.com/products/tecio-library/>`_ library is required.
If not, `Tecplot <https://www.tecplot.com/>`_ output file is written in ASCII format.

1. `TecIO` == 2014

For the colored LU-SGS scheme, the ``networkx`` package can be optionally used to perform graph coloring. If ``networkx`` is available, it is used in place of coloring based on ``scipy.sparse`` utilities.

1. `networkx` > 3.0
 
For RANS simulations, distances from wall boundaries must be computed. The ``pykdtree`` package can be optionally used to accelerate this process via KD-tree searches. If ``pykdtree`` is available, it is used in place of distance computations based on ``scipy.sparse`` utilities.

1. `pykdtree` >= 1.3
