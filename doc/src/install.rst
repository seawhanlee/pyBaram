*************
Introduction
*************

Overview
========

pyBaram
-------
pyBaram is an open-source, Python-based software designed to solve compressible flow using the finite volume method on unstructured grids. 'Baram' translates to 'Wind' in Korean. The software is tailored for solving compressible inviscid flow, laminar flow, and turbulent flow through the use of RANS (Reynolds Averaged Navier-Stokes) models. All the code is written in Python, and hybrid parallel simulations are implemented using high-performance Python packages.


*************
Installation
*************

pyBaram |version| can be obtained from the `repository <https://gitlab.com/aadl_inha/PyBaram>`_.
Currently, ``pyBaram`` supports only Linux system or WSL (Windows Subsystem Linux).

Quick start
===========
With `Anaconda <https://www.anaconda.com/>`_ (or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_) python distribution, you can readily install pyBaram.

1. Make a new environment and activate it::

    user@Computer ~/pyBaram$ conda create -n pybaram
    user@Computer ~/pyBaram$ conda activate pybaram

2. Install Python packages::

    user@Computer ~/pyBaram$ conda install numpy numba h5py metis
    user@Computer ~/pyBaram$ conda install -c conda-forge mpi4py cgns

3. Obtain ``pyBaram`` release version from `release page <https://gitlab.com/aadl_inha/PyBaram/-/releases>`_ and install it::

    user@Computer ~/pyBaram$ pip install pybaram-0.X.Y-py3-none-any.whl


Compile from source
===================
You can install this code using ``setup.py``::

    user@Computer ~/pyBaram$ pip install -e .

It is recommended to use ``virtualenv`` or ``conda`` to create a separate environment.

Dependencies
------------
pyBaram |version| requires python 3.7+ and following python packages.

1. `numpy` >= 1.10
2. `numba` >= 0.5
3. `h5py` >= 2.6
4. `mpi4py` >= 2.0
5. `tqdm` >= 4.0

In order to convert the mesh with CGNS format, CGNS library is required.

1. `CGNS` >= 3.4

To partitioning the mesh for parallel computation, `METIS` library is required.

1. `METIS` >= 5.1

To convert a solution to `Tecplot <https://www.tecplot.com/>`_ binary format, `TecIO <https://www.tecplot.com/products/tecio-library/>`_ library is required.
If not, `Tecplot <https://www.tecplot.com/>`_ output file is written in ASCII format.

1. `TecIO` == 2014

For serial LU-SGS scheme, `scipy.sparse` package is optionally required to re-order mesh with reverse Cuthill-McKee algorithm.
This reordering reduces the bandwidth of the implicit operation matrix. If `scipy` is not found, the same ordering as the mesh file is used.

1. `scipy` >= 1.0

For coloring LU-SGS scheme, `networkx` and `scipy.sparse` packages are optionally required to utilize greedy coloring algorithm in `netowrkx`.
If these packages are not found, pure python greedy algorithm is used.

For RANS computations, calculating the distances from the wall boundary is necessary. 
The `pykdtree` or `scipy.sparse` package can be optionally used 
to perform these calculations more efficiently using the KD-tree search algorithm. 
If neither package is available, the distances can still be computed via a brute-force approach.

1. `pykdtree` >= 1.3

2. `scipy` >= 1.6