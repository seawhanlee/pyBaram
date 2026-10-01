# Introduction

## Overview

pyBaram is an open-source Python compressible-flow solver using finite volumes
on unstructured grids. It supports inviscid, laminar, and turbulent flows,
including Reynolds-averaged Navier-Stokes (RANS) models and parallel execution.

This documentation describes the
[seawhanlee/pyBaram fork](https://github.com/seawhanlee/pyBaram) of
[aadl_inha/pyBaram](https://gitlab.com/aadl_inha/pyBaram). See
{doc}`fork_additions` for fork-specific features and the
[upstream documentation](https://aadl_inha.gitlab.io/pyBaram/) for the original
project.

# Installation

pyBaram {{ version }} requires Python 3.9 or newer. Linux, Windows, and macOS can
be used when the necessary third-party shared libraries are available. Shell
completion described below is supported for Bash and Zsh on Linux and macOS.

## Quick start with Conda

Conda is recommended because it installs MPI and the compiled scientific
packages together. Create and activate an environment:

```bash
conda create -n pybaram -c conda-forge python=3.12 numpy scipy numba h5py mpi4py rich 'argcomplete>=3.6.3,<3.7' pip
conda activate pybaram
```

For CGNS mesh import and MPI mesh partitioning, also install the native libraries:

```bash
conda install -c conda-forge cgns metis
```

Install from this fork's source:

```bash
git clone https://github.com/seawhanlee/pyBaram.git
cd pyBaram
python -m pip install .
pybaram --help
```

For editable development, use `python -m pip install -e .` instead.
Installation installs the Python dependencies declared in `pyproject.toml`;
it does not modify your shell configuration.

## Install a release wheel

Download an available wheel from the
[fork's GitHub releases](https://github.com/seawhanlee/pyBaram/releases),
then install that downloaded file in the activated environment:

```bash
python -m pip install /path/to/downloaded/pybaram-X.Y.Z-py3-none-any.whl
```

Replace `X.Y.Z` with the downloaded release version. Source installation
provides the current checkout; a previously published wheel may have fewer
features. Installing through pip also installs `rich` and `argcomplete`.

## Pip-only environments

A Python virtual environment can also be used. Install an MPI runtime and its
development libraries through your operating system first so `mpi4py` can
build or load. For example, on Ubuntu:

```bash
sudo apt-get install build-essential libopenmpi-dev openmpi-bin python3-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

Run these commands from a checkout of this fork. CGNS and METIS shared
libraries still need to be installed separately when those features are used.

## Dependencies

The required Python packages are:

- `numpy >= 1.10`
- `numba >= 0.5`
- `scipy >= 1.6`
- `h5py >= 2.6`
- `mpi4py >= 2.0`
- `rich >= 13.0`: the fork's inline terminal progress display.
- `argcomplete >= 3.6.3, < 3.7`: the fork's Bash and Zsh completion.

The last two packages support this fork's CLI additions. They are required
package dependencies, including when progress output is disabled. Document
building has separate dependencies; see {doc}`fork_additions`.

### Optional Python packages

1. `petsc4py`
2. `graph-tool`
3. `pykdtree >= 1.3`

The `petsc4py` package is required only for the `petsc` and `petsc-rank`
implicit relaxation methods. [PETSc](https://petsc.org/) must use real,
double-precision scalars.
The `petsc` method uses a distributed PETSc communicator, while
`petsc-rank` creates an independent `PETSc.COMM_SELF` solver on each MPI
rank.

The `graph-tool` package accelerates rank coloring during mesh import and
partitioning for the colored LU-SGS and colored block LU-SGS methods. When it
is unavailable, pyBaram uses its built-in sequential Python implementation.

The `pykdtree` package accelerates wall-distance searches for RANS
simulations. When it is unavailable, pyBaram uses its SciPy-based
implementation.

### Native libraries

The `CGNS >= 3.4` library is required to import CGNS meshes, and
`METIS >= 5.1` is required to partition meshes for parallel computation.

On Windows, `pyBaram` requires a METIS dynamic library (`metis.dll` or
`libmetis.dll`); a static or import `.lib` file alone cannot be loaded by
`ctypes`. If the conda package does not provide a METIS DLL, install a
prebuilt METIS DLL or build METIS as a shared library.

The [TecIO](https://www.tecplot.com/products/tecio-library/) 2014 library is
required for binary Tecplot output. Without TecIO, pyBaram writes Tecplot
output in ASCII format.

`pyBaram` loads CGNS, METIS, and TecIO through `ctypes`. The library search
path can be extended with the `PYBARAM_LIB_PATH` environment variable. Use
`:` to separate paths on Linux and macOS, and `;` on Windows.

For example, on Linux or macOS:

```
user@Computer ~/pyBaram$ export PYBARAM_LIB_PATH=/path/to/cgns/lib:/path/to/metis/lib
```

On Windows:

```
C:\> set PYBARAM_LIB_PATH=C:\path\to\cgns\bin;C:\path\to\metis\bin
```

When running inside a conda environment, `pyBaram` also searches common conda
library directories such as `Library\bin` on Windows.

#### CUDA support

The optional CUDA backend requires an NVIDIA CUDA-capable GPU and a CUDA driver
supported by the installed version of Numba.

(shell-completion)=

## Shell completion

After installation, activate the environment containing `pybaram` and
`register-python-argcomplete`. Register completion in your current Bash shell:

```bash
conda activate pybaram
eval "$(register-python-argcomplete --no-defaults pybaram)"
```

For Zsh, initialize its completion system first unless your shell configuration
already does so:

```zsh
conda activate pybaram
autoload -Uz compinit
compinit
eval "$(register-python-argcomplete --no-defaults pybaram)"
```

For persistent registration, add the registration line to `~/.bashrc` or
`~/.zshrc` after your environment activation and, for Zsh, after `compinit`.
If you activate the environment manually after starting a shell, register
completion afterward. pyBaram installation does not edit these files.

Examples of Tab completion:

```text
pybaram re<Tab>                                   # restart
pybaram run mesh.pbrm config.ini --backend c<Tab>  # cpu or cuda
pybaram run mesh.pbrm conf<Tab>                    # .ini files/directories
pybaram sweep mesh.pbrm config.ini --out study<Tab> # directories
```

Completion suggests commands, flags, choices, and paths. Input suggestions
are filtered by file type; output filenames can be entered freely. Numeric
values and surface names have no value suggestions. Completion does not read
mesh contents or start simulations. Registration targets the `pybaram` command
on `PATH`; it does not register `python -m pybaram` or MPI launcher commands.

If completion is unavailable, check the active commands and register again:

```bash
command -v pybaram
command -v register-python-argcomplete
python -m pip show argcomplete
eval "$(register-python-argcomplete --no-defaults pybaram)"
```

Both commands should belong to your active Python environment. Confirm that
registration succeeds, that Zsh's `compinit` ran first, and that you are
using the `pybaram` command directly.
