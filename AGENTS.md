# Repository Guidelines

## Project Structure & Module Organization

pyBaram is a Python compressible-flow solver using finite volumes on unstructured grids. Source code lives in `pybaram/`: `solvers/` contains flow and turbulence models, `backends/` provides CPU/CUDA execution, and `integrators/` implements time-stepping schemes. `api/` handles simulations, sweeps, progress, and cancellation; `__main__.py` defines the CLI. Mesh readers, output writers, and runtime plugins have separate packages.

`tests/` contains regression tests. `examples/` holds case configurations (`.ini`) and meshes (`.cgns`). Sphinx documentation lives in `doc/src/`, including assets under `_static/`. Packaging is configured in `pyproject.toml`.

## Build, Test, and Development Commands

Use Python 3.9 or newer. Prefer the README's Conda environment setup to obtain MPI and compiled scientific dependencies.

- `python -m pip install -e .`: install this checkout for editable development.
- `pybaram --help`: inspect CLI commands and verify installation.
- `python -m unittest discover -s tests -v`: run the regression suite.
- `pybaram run mesh.pbrm config.ini --backend cpu --ui none`: run a prepared case without progress output; import example meshes first with `pybaram import`.
- `python -m pip install build` followed by `python -m build --wheel`: produce a release wheel in `dist/`.
- `make -C doc html`: build documentation after installing Sphinx and the extensions/theme listed in `doc/src/conf.py`.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions and variables, and `PascalCase` classes. Follow nearby code's import ordering, quoting, and line wrapping. Keep numerical kernels consistent with the existing backend conventions. No formatter or linter is configured; avoid unrelated formatting changes.

## Testing Guidelines

Tests use standard-library `unittest` and `unittest.mock`. Name files `test_*.py` and methods `test_*`. Add focused regression tests for changed behavior, especially CLI parsing, simulation cleanup, MPI cancellation, and sweep output. No coverage threshold is configured. For numerical changes, also validate a representative example and report the configuration and observed results.

## Commit & Pull Request Guidelines

Recent commits use short, imperative subjects such as “Add keyboard cancellation” and “Bump version to 0.12.2”; follow that pattern. Keep commits focused. PR descriptions should explain the behavior change, link relevant issues, and record validation commands/results. Include terminal captures for visible progress changes. Update documentation for user-facing changes. Release tags `vX.Y.Z` must match `pybaram/_version.py`, as checked by the release workflow.

## Versioning & Required Commits

For major feature additions, increment the version by `0.1.0` and reset the patch component (e.g., `0.13.2` → `0.14.0`). For minor additions and bug fixes, increment by `0.0.1` (e.g., `0.13.0` → `0.13.1`). Update `pybaram/_version.py` and relevant version references together. After every feature addition, run appropriate checks and create a Git commit containing the feature and version updates. Exclude unrelated user changes from the commit.
