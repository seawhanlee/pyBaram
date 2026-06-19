pyBaram
========

Fork Notice
-----------
This repository is a fork of the original pyBaram project maintained at
[aadl_inha/pyBaram](https://gitlab.com/aadl_inha/pyBaram). The upstream project
is the authoritative source for the original solver, documentation, authorship,
and licensing history.

This fork is maintained at [seawhanlee/pyBaram](https://github.com/seawhanlee/pyBaram)
and may include changes that are not present upstream, such as release automation
and terminal UI improvements. When citing pyBaram or looking for the original
project context, refer to the upstream project and the paper listed below.

Overview
---------
pyBaram is an open-source, Python-based software designed to solve compressible flows using the finite volume method on unstructured grids. 'Baram' translates to 'Wind' in Korean. The software supports the simulation of compressible inviscid, laminar, and turbulent flows based on the Reynolds-averaged Navier-Stokes (RANS) models. All the code is written in Python, and hybrid parallel simulations are implemented using high-performance Python packages.

Examples
---------
Examples of using pyBaram are available in the examples directory. Currently available examples includes:

- 3D Inviscid spherical explosion problem

- 2D transonic turbulent flow over RAE2822 airfoil

- 3D transonic turbulent flow over ONERA M6 wing

- 3D supersonic turbulent flow around HB-2 model

Documentation
-------------
Information on the installation, usage, and implementation of pyBaram can be found in the [documentation](https://aadl_inha.gitlab.io/pyBaram/).

Reference
---------
[pyBaram: Parallel compressible flow solver in high-performance Python for teaching and research, SoftwareX, 2022](https://doi.org/10.1016/j.softx.2022.101272)

Authors
--------
See the AUTHORS file.

License
---------
pyBaram is released under the New BSD License.
