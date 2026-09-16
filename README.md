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

Upstream synchronization
------------------------
Fork version `0.12.1` incorporates upstream [v0.8.0](https://gitlab.com/aadl_inha/pyBaram/-/tags/v0.8.0)
(commit `a423cfa68b5d4053c8e5fee605d2a54076b54daf`), preserving the Rich CLI,
CLI progress options, and resumable AOA sweeps. New upstream features include
the CUDA backend, solver improvements, and rank-ordered/colored mesh layouts.
Use `--backend cpu` (default) or `--backend cuda` with `run` and `restart`;
AOA sweeps continue to use the CPU backend. See the user guide for mesh layout
requirements for implicit schemes.

Overview
---------
pyBaram is an open-source, Python-based software designed to solve compressible flows using the finite volume method on unstructured grids. 'Baram' translates to 'Wind' in Korean. The software supports the simulation of compressible inviscid, laminar, and turbulent flows based on the Reynolds-averaged Navier-Stokes (RANS) models. All the code is written in Python, and hybrid parallel simulations are implemented using high-performance Python packages.

Installation
------------
pyBaram requires Python 3.9 or newer. It depends on scientific Python packages
including `numpy`, `scipy`, `numba`, `h5py`, `mpi4py`, and `rich`.

The recommended installation method is Conda because it can install Python,
MPI, and the compiled scientific dependencies together in one environment:

```bash
conda create -n pybaram -c conda-forge \
  python=3.11 numpy scipy numba h5py mpi4py rich pip
conda activate pybaram
```

The previously published `0.10.0` release wheel remains available:

```bash
python -m pip install \
  https://github.com/seawhanlee/pyBaram/releases/download/v0.10.0/pybaram-0.10.0-py3-none-any.whl
```

To use version `0.12.1` from this checkout, create and activate the same Conda
environment first, then install from source:

```bash
git clone https://github.com/seawhanlee/pyBaram.git
cd pyBaram
python -m pip install .
```

For editable development installs, use:

```bash
python -m pip install -e .
```

Pip-only virtual environments can work, but they require MPI development
libraries to be installed separately before `mpi4py` can build or load
correctly. Conda is the safer default for most users.

Verify the command-line entry point:

```bash
pybaram --help
```

Progress display
----------------
All simulations use Rich progress output by default, including `run`, `restart`,
`sweep`, Python API calls, and MPI execution. The display stays inline in the
terminal; it does not open a full-screen interface.

```bash
pybaram run mesh.pbrm config.ini
pybaram restart mesh.pbrm solution.pbrs --ui rich
mpirun -n 2 pybaram run partitioned-mesh.pbrm config.ini
```

Available progress modes are:

- `rich` (default): progress, iteration/time, residual, CFL, and available solver status
- `none`: disable progress output

The former `tui` and `tqdm` option values are no longer accepted. Remove those
options or replace them with `--ui rich`. Python APIs accept `ui="rich"` or
`ui="none"` as well.

Only MPI rank 0 renders progress. On Linux, a local MPI launcher’s terminal is
also detected when MPI forwards rank output through pipes. Interactive terminals
refresh live; redirected
output and other non-terminal streams receive the final Rich status without
terminal control sequences. Sweeps print a status at the end of each executed
AOA case. Use `--ui none` to suppress this output.

AOA Sweep
---------
This fork adds an AOA sweep command for running the same mesh and base
configuration across multiple angles of attack. The command modifies
`[constants] aoa` for each case, so existing expressions such as
`u = uf*cos(aoa/180*pi)` and force-direction definitions automatically update.

Run explicit AOA values:

```bash
pybaram sweep mesh.pbrm config.ini --aoa 0,2,4
```

Run an inclusive range:

```bash
pybaram sweep mesh.pbrm config.ini --aoa-range -2 6 2
```

By default, sweep results are written under `sweep-aoa/`, with one directory per
AOA value. Positive values use names such as `sweep-aoa/aoa1/` and
`sweep-aoa/aoa2/`; negative values use names such as `sweep-aoa/aoan1/` and
`sweep-aoa/aoan2/`. Each case directory contains the resolved `config.ini` used
for that run and the normal pyBaram output files. A
`sweep-aoa/sweep.csv` file summarizes the final row from each `force_*.csv`
file so aerodynamic coefficient trends can be compared directly.

Use a custom output directory or progress mode with:

```bash
pybaram sweep mesh.pbrm config.ini --aoa 0,2,4 --out aoa-study --ui rich
```

For sweeps, `--ui rich` adds a sweep progress bar above the normal solver
progress display. It shows the number of completed AOA cases and the angle
currently running while preserving realtime per-case solver status. The right
side of the CLI lists each target AOA and its latest/final residual so completed
cases can be compared while the sweep continues.

If a case directory already exists and is not empty, the sweep stops rather than
appending to old CSV files. Use `--overwrite` only when you intentionally want
to replace existing case directories:

```bash
pybaram sweep mesh.pbrm config.ini --aoa 0,2,4 --out aoa-study --overwrite
```

Use `--resume` to continue a previous sweep without rerunning non-empty case
directories. Existing cases are marked as `skipped` in `sweep.csv`, new cases
are marked as `complete`, and the summary file is updated after each case:

```bash
pybaram sweep mesh.pbrm config.ini --aoa 0,2,4 --out aoa-study --resume
```

Examples
---------
Examples of using pyBaram are available in the examples directory. Currently available examples includes:

- 3D Inviscid spherical explosion problem

- 2D transonic turbulent flow over RAE2822 airfoil

- 2D unsteady laminar flow around a circular cylinder

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
