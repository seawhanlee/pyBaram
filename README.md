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

Terminal UI
-----------
This fork adds an optional terminal UI for monitoring simulations. The default
progress display is still `tqdm`, so existing commands continue to work without
changes.

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

The TUI requires the `rich` Python package, which is included in this fork's
runtime dependencies. In MPI runs, only rank 0 renders the progress display.

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

For sweeps, `--ui tui` shows one sweep-level dashboard with the number of
completed AOA cases and the angle currently running. Individual solver progress
bars are disabled inside each case so the sweep display remains stable.

If a case directory already exists and is not empty, the sweep stops rather than
appending to old CSV files. Use `--overwrite` only when you intentionally want
to replace existing case directories:

```bash
pybaram sweep mesh.pbrm config.ini --aoa 0,2,4 --out aoa-study --overwrite
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
