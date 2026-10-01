# Fork additions

## About this fork

This documentation describes the **seawhanlee/pyBaram fork** of
[aadl_inha/pyBaram](https://gitlab.com/aadl_inha/pyBaram). The original
project is maintained by AADL, Inha University. Its
[documentation](https://aadl_inha.gitlab.io/pyBaram/), authorship, and
licensing history remain the reference for the original solver. This fork
retains the New BSD License; see the
[AUTHORS](https://github.com/seawhanlee/pyBaram/blob/main/AUTHORS) and
[LICENSE](https://github.com/seawhanlee/pyBaram/blob/main/LICENSE) files.
When citing the solver, use the original
[SoftwareX paper](https://doi.org/10.1016/j.softx.2022.101272).

The fork incorporates upstream v0.8.0 (commit
`a423cfa68b5d4053c8e5fee605d2a54076b54daf`). CUDA support, solver
improvements, and rank-ordered/colored mesh layouts came from that upstream
update. The features below are fork additions. Their detailed command syntax
is also covered in {doc}`user_guide`.

## Inline progress

`run`, `restart`, `sweep`, and the simulation Python APIs default to
Rich progress output. It stays inline in the terminal and shows iteration or
simulation time, residuals, CFL, and available solver status, alongside elapsed
time, iteration rate, and estimated time remaining (ETA). ETA is initially
shown as `estimating` until sufficient progress is available.

```bash
pybaram run mesh.pbrm config.ini --ui rich
pybaram restart mesh.pbrm solution.pbrs --ui none
mpirun -n 2 pybaram run partitioned-mesh.pbrm config.ini
```

Only MPI rank 0 displays progress. The mesh must be partitioned for the number
of MPI ranks. On Linux, pyBaram also detects the terminal of a local MPI
launcher when rank output is forwarded through pipes. Redirected output
receives a final status without live terminal control sequences; sweeps emit
one final status per executed case. `--ui none` suppresses progress output.
Python callers can use `ui="rich"` or `ui="none"`. The former `tui` and
`tqdm` option values are no longer accepted.

## Keyboard cancellation

Press `q` during a simulation to stop at the next iteration boundary. The
request is shared with all MPI ranks and prevents remaining sweep cases from
running. With an MPI launcher that forwards line-buffered input, press `q`
followed by Enter. Existing output files are retained, but cancellation does
not force a new solution checkpoint. Restart from a solution checkpoint that
has already been written. Python callers can catch
`pybaram.api.stop.SimulationStopped`.

## Angle-of-attack sweeps

The `sweep` command runs one mesh and base configuration at multiple angles
of attack by replacing `[constants] aoa` for each case. Velocity and force
direction expressions that reference `aoa` are re-evaluated.

```bash
pybaram sweep mesh.pbrm config.ini --aoa 0,2,4
pybaram sweep mesh.pbrm config.ini --aoa-range -2 6 2 --out aoa-study
pybaram sweep mesh.pbrm config.ini --aoa 0,2,4 --out aoa-study --resume
```

The default output directory is `sweep-aoa/`. Each angle has a directory
such as `aoa2/` or `aoan2/` containing its resolved `config.ini` and
normal solver outputs. `sweep.csv` summarizes the last row of each
`force_*.csv` file and is updated after each case. The Rich display adds
sweep progress and a table of target angles and their latest/final residuals.
Sweeps use the CPU backend.

Existing non-empty case directories are rejected by default. `--resume`
skips those directories and marks them as `skipped`; it does not verify
that those cases finished or restart an interrupted case. New completed cases
are marked as `complete`. `--overwrite` replaces existing case
directories and should be used only when their previous results can be
discarded. `--resume` and `--overwrite` are mutually exclusive.

## Shell completion

The fork includes Bash and Zsh completion using `argcomplete`. It suggests
commands, flags, choices, and relevant input paths without reading meshes or
starting simulations. Input paths are filtered by file type, while output
filenames can be entered freely. See {ref}`shell-completion` for registration,
persistent shell configuration, and troubleshooting.

## Release wheels

The fork's GitHub release workflow builds wheels from version tags and checks
that each tag matches the package version. Available wheels are listed on the
[fork release page](https://github.com/seawhanlee/pyBaram/releases).

## Building and publishing documentation

The documentation uses Sphinx, MyST-Parser (Markdown), PyData Sphinx Theme,
and sphinx-autodoc2 for API references. Documentation builds require Python
3.11 or newer; Python 3.12 is recommended for development. The solver itself
continues to support Python 3.9 or newer.

Install the solver as described in {doc}`install`, then install the documentation
packages from the repository root. For the development environment:

```bash
conda activate pybaram-dev
python -m pip install -e .
python -m pip install -r doc/requirements.txt
make -C doc html SPHINXOPTS="-n -W --keep-going"
```

The build requires Graphviz (`dot`) for diagrams and LaTeX plus `dvipng`
for equation images. On Ubuntu, install them with:

```bash
sudo apt-get install graphviz texlive-latex-extra dvipng
```

Open `doc/build/html/index.html` to preview the result. The Documentation
GitHub Actions workflow validates pull requests and publishes successful
builds from `main` to
[GitHub Pages](https://seawhanlee.github.io/pyBaram/). It also supports a
manual run on `main`. In the repository's **Settings > Pages**, set
**Build and deployment > Source** to **GitHub Actions**. Build packages are
only required when generating documentation, not when using the solver.

### Editing documentation and API references

Write pages as MyST Markdown in `doc/src/*.md`. MyST supports Sphinx
cross-references, fenced directives, equations, tables, and bibliography
citations. The PyData theme provides search, light/dark appearance, and links
to edit each page on GitHub.

API references are placed in the guides with `autodoc2-object`, for example:

````markdown
```{autodoc2-object} pybaram.api.simulation.run
```
````

autodoc2 statically analyses `pybaram/`; it does not import the solver to
extract API signatures and docstrings. Existing Python docstrings remain
RST and are parsed with the explicit `rst` setting in `doc/src/conf.py`.
The inheritance diagrams still import their classes and require the solver's
runtime dependencies. Keep manually selected API sections in the guides;
automatic generation of a page for every module is disabled.

The local `doc/_ext/autodoc2_context.py` extension corrects autodoc2 0.5's
inline class context so API anchors and source links keep their existing
addresses. Run its regression check with
`python -m unittest discover -s doc/tests -v`; the documentation workflow
runs this check before building the site.
