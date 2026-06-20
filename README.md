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

Installation
------------
pyBaram requires Python 3.9 or newer. It depends on scientific Python packages
including `numpy`, `scipy`, `numba`, `h5py`, `mpi4py`, `tqdm`, `rich`, and
`textual`. On Linux, install the MPI development libraries before installing
pyBaram so `mpi4py` can build or load correctly.

Recommended isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the latest release wheel from this fork:

```bash
python -m pip install \
  https://github.com/seawhanlee/pyBaram/releases/download/v0.10.0/pybaram-0.10.0-py3-none-any.whl
```

Or install from a local checkout:

```bash
git clone https://github.com/seawhanlee/pyBaram.git
cd pyBaram
python -m pip install .
```

For editable development installs:

```bash
python -m pip install -e .
```

Verify the command-line entry point:

```bash
pybaram --help
pybaram tui
```

Terminal UI
-----------
This fork adds a full-screen terminal UI experience. Use the launcher when you
want a guided, keyboard-first workflow for pyBaram commands instead of
memorizing CLI arguments:

```bash
pybaram tui
```

The launcher is built on Textual and opens persistent panes for:

- the current working directory, so you always know where pyBaram is browsing;
- a navigable local file list with typed filtering/search for path completion;
- workflow and field selection for fresh runs, restarts, AOA sweeps, mesh import,
  mesh partitioning, and solution export;
- an exact command preview before execution;
- a live output pane that streams command output from a subprocess boundary.

Core shortcuts include `j`/`k` for file navigation, `Enter` to open a directory or
assign the selected file to the active field, `Backspace` for the parent
directory, `w` to cycle workflows, `f` to cycle fields, `c` to cycle choices
(such as AOA values vs range), `r` to run the previewed command, `Tab` to move
focus, and `q` to quit. The TUI intentionally does not
edit solver `.ini` files, redesign solver behavior, create project databases, or
act as a remote file manager.

The TUI requires the `textual` Python package, which is included in this fork's
runtime dependencies. If the package is unavailable, `pybaram tui` reports a
clear missing-dependency message rather than failing with an import traceback.

You can also opt into the live dashboard directly from normal commands. The
default progress display is still `tqdm`, so existing commands continue to work
without changes.

Use the Rich-based terminal dashboard with `--ui tui`:

```bash
pybaram run mesh.pbrm config.ini --ui tui
```

For restarted simulations:

```bash
pybaram restart mesh.pbrm solution.pbrs --ui tui
```

Available progress modes are:

- `tqdm`: default progress bar
- `tui`: Rich terminal dashboard with progress, iteration/time, residual, CFL,
  and related solver status where available
- `none`: disable progress output, useful for batch jobs and log files

For non-interactive execution, use:

```bash
pybaram run mesh.pbrm config.ini --ui none
```

In MPI runs, only rank 0 renders the progress display.

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
pybaram sweep mesh.pbrm config.ini --aoa 0,2,4 --out aoa-study --ui tui
```

For sweeps, `--ui tui` adds a sweep progress bar above the normal solver
progress display. It shows the number of completed AOA cases and the angle
currently running while preserving realtime per-case solver status. The right
side of the TUI lists each target AOA and its latest/final residual so completed
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
