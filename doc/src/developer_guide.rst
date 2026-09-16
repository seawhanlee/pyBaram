***************
Developer Guide
***************

Overview of Code Structure
==========================

Start
-----
The ``pybaram`` console command is linked to :mod:`pybaram.__main__`. The
``run`` and ``restart`` subcommands load the native mesh, solution, and INI
files in ``process_run`` or ``process_restart``, then call
:func:`pybaram.api.simulation.run` or
:func:`pybaram.api.simulation.restart`. The shared ``_common`` helper selects
the backend and integrator before starting the simulation loop.

Integrators
-----------
The Integrator object conducts time integration of the discretized equations. When the ``integrator`` is initiated, it invokes the `system` class in the :mod:`pybaram.solvers` module to compute the right-hand side term of the FVM. Additionally, plugins are invoked by this integrator object for post-processing.

``pybaram`` selects the concrete integrator from ``mode`` and ``stepper`` in
``[solver-time-integrator]``. The mode determines the base class:
:mod:`pybaram.integrators.unsteady` for explicit physical-time integration,
:mod:`pybaram.integrators.steady` for steady iteration, and
:mod:`pybaram.integrators.dts` for unsteady dual-time stepping. The selected
class then constructs the kernels required by its stages or relaxation
solver.

Unsteady Integrators
********************
For ``mode = unsteady``, the physical solution is advanced directly by an
explicit time-integration scheme. The ``BaseUnsteadyIntegrator`` controls the
physical-time list, restart time, time-step controller, and plugin callbacks.
Concrete steppers assemble their stage kernels in ``construct_stages``.

.. admonition:: TVD-RK3
   :class: dropdown

    .. autoclass:: pybaram.integrators.unsteady.TVDRK3
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:


Steady Integrators
******************
For ``mode = steady``, the integrator iterates until either ``max-iter`` is
reached or the selected residual drops below ``tolerance``. Explicit steady
steppers use local pseudo time steps and Runge-Kutta-like stage kernels. The
implicit steady steppers are thin wrappers around the shared relaxation
solvers in :mod:`pybaram.integrators.relaxation`.

.. admonition:: 5-stage Runge-Kutta
   :class: dropdown

    .. autoclass:: pybaram.integrators.steady.FiveStageRK
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

.. admonition:: LU-SGS
   :class: dropdown

    .. autoclass:: pybaram.integrators.steady.LUSGS
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

.. admonition:: Colored LU-SGS
   :class: dropdown

    .. autoclass:: pybaram.integrators.steady.ColoredLUSGS
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

.. admonition:: Block LU-SGS
   :class: dropdown

    .. autoclass:: pybaram.integrators.steady.BlockLUSGS
      :members:
      :undoc-members:
      :inherited-members:
      :private-members:

.. admonition:: Colored Block LU-SGS
   :class: dropdown

    .. autoclass:: pybaram.integrators.steady.ColoredBlockLUSGS
      :members:
      :undoc-members:
      :inherited-members:
      :private-members:

.. admonition:: Distributed PETSc
   :class: dropdown

    .. autoclass:: pybaram.integrators.steady.PETSc
      :members:
      :undoc-members:
      :inherited-members:
      :private-members:

.. admonition:: Rank-local PETSc
   :class: dropdown

    .. autoclass:: pybaram.integrators.steady.PETScRank
      :members:
      :undoc-members:
      :inherited-members:
      :private-members:


Dual-Time Stepping Integrators
******************************
For ``mode = unsteady-dts``, physical time is advanced with a fixed ``dt``,
while each physical step is solved by pseudo-time sub-iterations. The
``BaseDTSIntegrator`` manages the physical-time loop, BDF solution history,
pseudo-time CFL, sub-iteration convergence, and restart indices. The concrete
BDF classes define the BDF coefficients and assemble the source term for the
pseudo-steady problem.

Higher-order BDF DTS steppers start with lower-order formulas until enough
physical-time solution history is available. Thus ``bdf2`` uses BDF1 for the
first physical step and then BDF2, while ``bdf3`` uses BDF1, then BDF2, and
then BDF3. When a restart file records the completed physical-step counter,
this startup order is preserved across restart.

The pseudo-time solve itself is not implemented in the DTS classes. Instead,
``BaseDTSIntegrator`` selects a relaxation solver from
``[solver-time-relaxation]`` and calls it at each sub-iteration.

.. admonition:: BDF1 Dual-Time Stepping
   :class: dropdown

    .. autoclass:: pybaram.integrators.dts.BDF1DTSIntegrator
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

.. admonition:: BDF2 Dual-Time Stepping
   :class: dropdown

    .. autoclass:: pybaram.integrators.dts.BDF2DTSIntegrator
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

.. admonition:: BDF3 Dual-Time Stepping
   :class: dropdown

    .. autoclass:: pybaram.integrators.dts.BDF3DTSIntegrator
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:


Relaxation Solvers
******************
Relaxation solvers are implemented in
:mod:`pybaram.integrators.relaxation` and are shared by steady implicit
integrators and dual-time stepping integrators. The ``get_relaxation`` helper
selects a subclass of ``BaseRelaxation`` from the configuration and attaches
the relaxation solver to the owning integrator.

For steady simulations, ``SteadyRelaxationIntegrator`` passes the steady
stepper name directly to ``get_relaxation``. For ``unsteady-dts``
simulations, ``BaseDTSIntegrator`` reads the relaxation method from
``[solver-time-relaxation]``. In both cases, the relaxation object builds the
LU-SGS or block LU-SGS kernels and its ``step`` method computes the
right-hand side, performs the relaxation sweeps, updates the solution, and
applies post-processing.

Scalar LU-SGS relaxation uses ``spectral-radius`` as the implicit operator.
Block LU-SGS relaxation uses ``approx-jacobian`` and can perform inner
sub-iterations for the block correction. These solvers operate on a rank-wide
mixed-element cell graph rather than separate element-type graphs. The graph
stores cell-to-face adjacency in CSR form so sweeps and implicit-operator
assembly share the same global rank numbering.

The serial LU-SGS and block LU-SGS variants require the RCM-based
``rank-order`` layout stored in ``.pbrm`` meshes. Colored variants require the
``rank-coloring`` layout stored in ``.pbrmc`` meshes and schedule rank-wide
cells through color barriers. Import and partition physically group cells by
their selected coloring (``greedy`` or ``smallest-last``); graph-tool is used
when available, with a Python implementation as the fallback.

Both distributed ``petsc`` and rank-local ``petsc-rank`` use the
``rank-order`` layout and therefore require ``.pbrm``. Explicit integrators do
not construct rank-wide implicit storage and can use either native mesh
format. For dual-time stepping, the relaxation solver selected in
``[solver-time-relaxation]`` supplies the layout requirement to the owning
integrator before the system is constructed.

The ``petsc`` relaxation constructs owned rows of one distributed block sparse
(BSR) matrix. Its global column IDs include cells across MPI interfaces, and a
collective KSP solves the complete-domain operator. ``petsc-rank`` uses the
same rank-cell packing and face assembly kernels but creates an independent
``PETSc.COMM_SELF`` BSR matrix and KSP per rank. Both paths reuse their fixed
sparsity patterns while rebuilding numerical values and right-hand sides at
each relaxation step.

.. admonition:: LU-SGS Relaxation
   :class: dropdown

    .. autoclass:: pybaram.integrators.relaxation.LUSGSRelaxation
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

.. admonition:: Colored LU-SGS Relaxation
   :class: dropdown

    .. autoclass:: pybaram.integrators.relaxation.ColoredLUSGSRelaxation
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

.. admonition:: Block LU-SGS Relaxation
   :class: dropdown

    .. autoclass:: pybaram.integrators.relaxation.BlockLUSGSRelaxation
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

.. admonition:: Colored Block LU-SGS Relaxation
   :class: dropdown

    .. autoclass:: pybaram.integrators.relaxation.ColoredBlockLUSGSRelaxation
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

.. admonition:: Distributed PETSc Relaxation
   :class: dropdown

    .. autoclass:: pybaram.integrators.relaxation.PETScGlobalRelaxation
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:

.. admonition:: Rank-local PETSc Relaxation
   :class: dropdown

    .. autoclass:: pybaram.integrators.relaxation.PETScRankRelaxation
        :members:
        :undoc-members:
        :inherited-members:
        :private-members:


The hierarchy of ``integrator`` class can be shown as below.

.. inheritance-diagram:: pybaram.integrators.unsteady.TVDRK3
                         pybaram.integrators.dts.BDF1DTSIntegrator
                         pybaram.integrators.dts.BDF2DTSIntegrator
                         pybaram.integrators.dts.BDF3DTSIntegrator
                         pybaram.integrators.steady.FiveStageRK
                         pybaram.integrators.steady.LUSGS
                         pybaram.integrators.steady.ColoredLUSGS
                         pybaram.integrators.steady.BlockLUSGS
                         pybaram.integrators.steady.ColoredBlockLUSGS
                         pybaram.integrators.steady.PETSc
                         pybaram.integrators.steady.PETScRank
    :parts: 1 


Solvers
-------
In the :mod:`pybaram.solvers` module, the governing equations and their spatial discretizations are implemented. For each submodule corresponding to governing equations, there are objects such as  ``system``, ``elements``, ``inters`` and ``vertex``.

System
*******
The ``system`` object, invoked from the ``integrator``,  initializes 
``elements``, ``inters`` and ``vertex`` objects by reading mesh and restarted solution, if available. These objects have a `construct_kernels` method to generate kernels for computing the right-hand side. Here, the ``rhside`` method schedules these kernels. To enhance efficiency, non-blocking communications and computations are overlapped. 
The class hierarchy of the ``system`` can be depicted as follows:

.. inheritance-diagram:: pybaram.solvers.ranskwsst.system
                         pybaram.solvers.ranssa.system
                         pybaram.solvers.navierstokes.system
                         pybaram.solvers.euler.system
    :top-classes: pybaram.solvers.base.system.BaseSystem
    :parts: 1 

|

* ``BaseSystem`` : initiates objects and generates kernels from these objects

* ``BaseAdvecSystem`` : `rhside` method for advection problems, such as Euler systems.

    .. admonition:: rhside for advection
      :class: dropdown

        .. automethod:: pybaram.solvers.baseadvec.system.BaseAdvecSystem.rhside

* ``BaseAdvecDiffSystem`` : `rhside` method for advection-diffusion problems,
  such as the Navier-Stokes system.

    .. admonition:: rhside for advection-diffusion
      :class: dropdown

        .. automethod:: pybaram.solvers.baseadvecdiff.system.BaseAdvecDiffSystem.rhside

* ``RANSSystem`` : initiates objects and generates kernels from these objects for RANS simulation


Elements
********
The ``elements`` object stores solution and other arrays. It also generates kernels, looping over elements. The class hierarchy can be depicted as follows:

.. inheritance-diagram:: pybaram.solvers.navierstokes.elements
                         pybaram.solvers.euler.elements
    :top-classes: pybaram.solvers.base.elements.BaseElements
    :parts: 1 

* ``BaseElements`` : defines geometry and related properties

* ``BaseAdvecElements`` : common kernels for finite volume method, allocation of arrays

* ``EulerElements`` : specific kernels for Euler equations

* ``NavierStokesElements`` : specific kernels for Navier-Stokes equations

* ``FluidElements`` : physics of compressible inviscid flow

* ``ViscousFluidElements`` : physics of viscous flow

|

For RANS simulation, class hierarchy can be depicted as follows:

.. inheritance-diagram:: pybaram.solvers.ranskwsst.elements
                         pybaram.solvers.ranssa.elements
                         pybaram.solvers.ranssaneg.elements
    :top-classes: pybaram.solvers.base.elements.BaseElements
    :parts: 1

* ``RANSElements`` : common kernels for RANS computation

* ``RANSSAElements`` : specific kernels for Spalart-Allmaras turbulence model

* ``RANSSANegElements`` : specific kernels for negative Spalart-Allmaras turbulence model

* ``RANSKWSSTElements`` : specific kernels for SST turbulence model

* ``RANSSAFluidElements`` : physics of Spalart-Allmaras turbulence model

* ``RANSSANegFluidElements`` : physics of negative Spalart-Allmaras turbulence model

* ``RANSKWSSTFluidElements`` : physics of SST turbulence model


Inters
*******
The ``inters`` objects generate kernels looping over interfaces. There are three types of interfaces: Internal, boundary, and MPI interfaces. The abstract classes for them can be depicted as follows:

.. inheritance-diagram:: pybaram.solvers.base.BaseIntInters
                         pybaram.solvers.base.BaseBCInters
                         pybaram.solvers.base.BaseMPIInters
    :top-classes: pybaram.solvers.base.inters.BaseInters
    :parts: 1

* ``BaseInters`` : computes geometrical properties and defines view to refer array in ``elements``

* ``BaseIntInters`` : abstract class for internal interface

* ``BaseBCInters`` : abstract class for physical boundary interface

* ``BaseMPIInters`` : abstract class for MPI boundary interface

|

The class hierarchy of internal interfaces can be depicted as follows:

.. inheritance-diagram:: pybaram.solvers.ranskwsst.inters.RANSKWSSTIntInters
                         pybaram.solvers.ranssa.inters.RANSSAIntInters
                         pybaram.solvers.ranssaneg.inters.RANSSANegIntInters
                         pybaram.solvers.navierstokes.inters.NavierStokesIntInters
                         pybaram.solvers.euler.inters.EulerIntInters
    :top-classes: pybaram.solvers.base.inters.BaseIntInters
    :parts: 1 

* ``BaseAdvecIntInters`` : common kernel to compute :math:`\Delta U_{fi}`

* ``BaseAdvecDiffIntInters`` : common kernel to compute :math:`\nabla U_f`

* ``EulerIntInters`` : kernel to compute inviscid flux

* ``NavierStokesIntInters`` : kernel to compute viscous flux

* ``RANSIntInters`` : kernel to compute RANS flux

* ``RANSSAInters`` : kernel to compute turbulent flux for Spalart-Allmaras turbulence model

* ``RANSSANegInters`` : kernel to compute turbulent flux for negative Spalart-Allmaras turbulence model

* ``RANSKWSSTInters`` : kernel to compute turbulent flux for SST turbulence model

The class hierarchy of physical boundary interfaces can be depicted as follows:

.. inheritance-diagram:: pybaram.solvers.ranskwsst.inters.RANSKWSSTBCInters
                         pybaram.solvers.ranssa.inters.RANSSABCInters
                         pybaram.solvers.ranssaneg.inters.RANSSANegBCInters
                         pybaram.solvers.navierstokes.inters.NavierStokesBCInters
                         pybaram.solvers.euler.inters.EulerBCInters
    :top-classes: pybaram.solvers.base.inters.BaseBCInters
    :parts: 1 

The overall structure and role of these classes are the same as internal interfaces. The  ``construct_bc`` method in ``BaseAdvecInters`` compiles the boundary condition function, and specific formulations are implemented in this class. For example, the hierarchy of boundary conditions for Euler equations can be depicted as follows:

.. inheritance-diagram:: pybaram.solvers.euler.inters.EulerSupOutBCInters
                         pybaram.solvers.euler.inters.EulerSlipWallBCInters
                         pybaram.solvers.euler.inters.EulerSupInBCInters
                         pybaram.solvers.euler.inters.EulerFarInBCInters
                         pybaram.solvers.euler.inters.EulerSubOutPBCInters
                         pybaram.solvers.euler.inters.EulerSubInvBCInters
                         pybaram.solvers.euler.inters.EulerSubInpttBCInters
                         pybaram.solvers.euler.inters.EulerSubOutMdotBCInters
    :top-classes: pybaram.solvers.euler.inters.EulerBCInters
    :parts: 1 

The class hierarchy of MPI interfaces can be depicted as follows:

.. inheritance-diagram:: pybaram.solvers.ranskwsst.inters.RANSKWSSTMPIInters
                         pybaram.solvers.ranssa.inters.RANSSAMPIInters
                         pybaram.solvers.ranssaneg.inters.RANSSANegMPIInters
                         pybaram.solvers.navierstokes.inters.NavierStokesMPIInters
                         pybaram.solvers.euler.inters.EulerMPIInters
    :top-classes: pybaram.solvers.base.inters.BaseMPIInters
    :parts: 1 

The overall structure and roles of these classes are the same as those of
internal interfaces.
MPI communication kernels are defined in ``BaseAdvecMPIInters``.

Vertex
*******
The ``vertex`` object generates kernel looping over vertex. The class hierarchy can be depicted as follows:

.. inheritance-diagram:: pybaram.solvers.baseadvec.vertex
    :parts: 1 

* ``BaseVertex`` : view to refer array in ``elements``

* ``BaseAdvecVertex`` : kernel to find extreme values at vertex

Plugins
-------
The ``plugin`` modules handle the post-processing after each iteration or a fixed number of iterations. The class hierarchy can be depicted as follows:

.. inheritance-diagram:: pybaram.plugins.stats
                         pybaram.plugins.writer
                         pybaram.plugins.force
                         pybaram.plugins.surfint
    :top-classes: pybaram.plugins.base.BasePlugin
    :parts: 1 

* ``StatsPlugin`` : collect statistics (time step or residual)

* ``WriterPlugin`` : write output file

* ``ForcePlugin`` : compute aerodynamic force coefficients

* ``SurfIntPlugin`` : compute integrated and averaged properties over boundary surface.

Backends
--------
The :mod:`pybaram.backends` module accelerates pure Python loops and manages
kernel execution. ``CPUBackend`` uses Numba for serial or threaded CPU
execution, while ``GPUBackend`` compiles and launches CUDA kernels and manages
host-device transfers. This module provides two main features: generating
kernels and handling execution data types.

Kernel Compilation
******************
In the ``integrators`` and ``solvers`` modules, kernels are defined as pure
Python functions. The CPU backend compiles them into serial or parallel loops
using :mod:`pybaram.backends.cpu.loop`. The CUDA backend translates the same
loop functions and compiles them as CUDA kernels using
:mod:`pybaram.backends.cuda.loop`.

Data Types for Execution
************************
Eight execution helper types are defined in :mod:`pybaram.backends.types`.
They include array and kernel wrappers, three MPI kernel wrappers that
coordinate packing, host-device copies, and sends for CUDA execution, and the
MPI request queue.

.. automodule:: pybaram.backends.types
    :members:

Core Variables
--------------
The name of the variable ``pyBaram`` may seem somewhat condensed. The table below provides a summary of mathematical symbols and the corresponding meanings of major arrays:

.. list-table:: Notation of Variables in `pyBaram`
   :widths: 15 15 45 25
   :header-rows: 1

   * - Name
     - Symbol
     - Meaning
     - Notes
   * - upts
     - :math:`\bar{U}_i`
     - array of cell-averaged state variable vector
     -     
   * - fpts
     - :math:`U_f^\pm`
     - array of state vectors at faces
     -    
   * - grad
     - :math:`\nabla U_i` 
     - array of gradient of the state variables
     -    
   * - lim
     - :math:`\phi_i` 
     - array of slope limiter
     -    
   * - dt
     - :math:`\Delta t` 
     - array of time step size
     -    
   * - vpts
     - 
     - array of minimum and maximum at each vertex
     -    
   * - vol
     - :math:`\Delta V_i` 
     - array of volume of cell
     -    
   * - mag_snorm
     - :math:`\Delta A_f`
     - array of area of face
     -    
   * - vec_snorm
     - :math:`n_f`
     - array of unit normal vector of face
     -    


Code Snippets Analysis
======================
Here, the methods for generating kernels and constructing MPI communications are explained with two sample code snippets.

Inviscid Flux Kernel
--------------------
Solver kernels are written once as backend-independent Python functions. As an
example, :meth:`EulerIntInters._make_flux
<pybaram.solvers.euler.inters.EulerIntInters._make_flux>` defines an inviscid
flux function whose first two arguments, ``i_begin`` and ``i_end``, delimit an
outer cell-face loop. It obtains local-array allocation and compiled flux
helpers from the active backend, then passes the function and its static arrays
to ``self.be.make_loop``.

The selected backend determines what happens next::

    EulerIntInters._make_flux
                |
                +-- CPU single   -> numba.jit(func)
                +-- CPU parallel -> parse_loop -> numba.prange -> numba.jit
                +-- CUDA         -> parse_loop_gpu -> cuda.grid -> cuda.jit

The CPU loop generators are implemented in
:mod:`pybaram.backends.cpu.loop`. ``make_serial_loop1d`` compiles the original
function directly with Numba. ``make_parallel_loop1d`` first calls
:func:`pybaram.backends.parse.parse_loop` to replace the outer ``range`` with
``numba.prange``, recreates the function with its captured closure variables,
and compiles it with ``parallel=True``.

.. automodule:: pybaram.backends.cpu.loop
    :members:
    :undoc-members:

The CUDA path is implemented by
:func:`pybaram.backends.cuda.loop.make_cuda_loop`. It uses
:func:`pybaram.backends.parse.parse_loop_gpu` to replace the outer loop with a
one-dimensional CUDA grid index and a bounds check. The parser also maps the
backend-neutral local allocator ``array`` to ``cuda.local.array`` and maps the
supported NumPy spellings ``np.sqrt``, ``np.abs``, ``np.tanh``, and ``np.exp``
to CUDA-compatible functions. The transformed function is recreated with its
closure variables and compiled with ``cuda.jit`` before being launched with
the configured block size and compute stream.

.. automodule:: pybaram.backends.cuda.loop
    :members: make_cuda_loop
    :undoc-members:

The following simplified example shows the outer-loop transformation. A solver
kernel is authored as::

    def comm_flux(i_begin, i_end, uf):
        for idx in range(i_begin, i_end):
            fn = array((nvars,), np.float64)
            # Compute and store the flux for face idx.

For CPU parallel execution, :func:`~pybaram.backends.parse.parse_loop`
rewrites the loop as::

    for idx in nb.prange(i_begin, i_end):
        ...

For CUDA execution, :func:`~pybaram.backends.parse.parse_loop_gpu` rewrites it
conceptually as::

    idx = cuda.grid(1)
    if idx < i_end:
        fn = cuda.local.array((nvars,), np.float64)
        ...

Functions without an outer loop, such as numerical-flux and Jacobian helpers,
are compiled as CUDA device functions. ``GPUBackend.compile`` uses
:func:`~pybaram.backends.parse.parse_simple_gpu` for symbol substitution
without adding a grid index.

.. automodule:: pybaram.backends.parse
    :members: parse_loop, parse_loop_gpu, parse_simple_gpu

These transformations deliberately support the kernel form used by pyBaram;
they are not a general Python-to-CUDA compiler. A loop kernel must expose its
iteration bounds as the first two arguments and contain an outer
``range(i_begin, i_end)`` loop that can be found in its source. Kernel source
must remain available through the standard-library ``inspect`` module, unless the caller supplies it
explicitly through ``src``. New NumPy operations or allocation spellings must
also be added to the parser before they can be used by CUDA kernels.

.. autoclass:: pybaram.solvers.euler.inters.EulerIntInters

  .. method:: _make_flux

The backend loop generator returns a callable together with the arrays bound
while the kernel was created. ``construct_kernels`` wraps these in
:class:`pybaram.backends.types.Kernel`, which retains the static arguments.
When the wrapper is called, it appends any runtime arguments, resolves
``ArrayBank`` objects to their active arrays, and invokes the compiled CPU or
CUDA kernel.

.. autoclass:: pybaram.solvers.euler.inters.BaseAdvecIntInters

  .. method:: construct_kernels

Non-blocking Send/Receive 
-------------------------
``pyBaram`` exploits the ``mpi4py`` package for MPI communication. Non-blocking communications are employed and overlapped with computing kernels. These methods are implemented in the ``MPIInters`` class.

In the ``construct_kernels`` method, non-blocking send and receive kernels,
along with their requests, are constructed using the ``_make_send`` and
``_make_recv`` methods. Buffers are passed to these methods, and the
``_sendrecv`` method is invoked. This method returns the ``start`` function.
When called with a ``Queue`` instance from ``rhside``, it registers the MPI
request with the queue and starts the non-blocking communication. The
communication is finalized when the queue's ``sync`` method is called.

.. autoclass:: pybaram.solvers.baseadvec.inters.BaseAdvecMPIInters

    .. method:: construct_kernels

    .. method:: _sendrecv
    
    .. method:: _make_send

    .. method:: _make_recv

.. autoclass:: pybaram.backends.types.Queue
    :noindex:

    .. method:: register

    .. method:: sync

CUDA Stream Overlap
*******************
For the CUDA backend, the ``construct_kernels`` method also creates ``pack`` and
``unpack`` kernels for transferring MPI-interface data between device arrays
and pinned host buffers. The ``pack`` kernel first gathers the interface data
into a device send buffer on the compute stream. A CUDA event then makes the
copy stream wait for packing to complete before starting the device-to-host
(D2H) transfer. The MPI send is started only after the host send buffer is
ready, while rank-local kernels can continue executing on the compute stream.

After the non-blocking MPI receive has completed, the ``unpack`` kernel reflects
the received buffer into backend-local storage. For CUDA, it first starts a
host-to-device (H2D) transfer on the copy stream. Another CUDA event makes the
compute stream wait for this transfer before the received values are used by
MPI-interface kernels. These event dependencies preserve the required ``pack
-> D2H -> MPI send`` and ``MPI receive -> H2D -> MPI-interface processing``
ordering without synchronizing the entire device.

When multiple node-local MPI ranks are used, the compute and copy operations
use separate CUDA streams, allowing rank-local computation to overlap with
host-device transfers and MPI communication. With a single node-local MPI
rank, both operations use the same stream and this stream-level overlap is
disabled.

The following simplified scheduling sequence illustrates this process::

    self.eles.compute_fpts()

    if self.mpiint:
        self.mpiint.recv(q)       # Post the persistent MPI receive
        self.mpiint.pack()        # Pack on the compute stream, then start D2H

    # These kernels can overlap with D2H on the copy stream.
    self.iint.compute_delu()
    self.bint.compute_delu()

    if self.mpiint:
        self.mpiint.send(q)       # Wait for D2H, then start the MPI send
        q.sync()                  # Wait for the MPI requests
        self.mpiint.unpack()      # Start H2D on the copy stream
        self.mpiint.compute_delu()


Module Index
============
The following modules form the main extension points described in this guide.

.. py:module:: pybaram.__main__
.. py:module:: pybaram.backends
.. py:module:: pybaram.integrators
.. py:module:: pybaram.integrators.unsteady
.. py:module:: pybaram.integrators.steady
.. py:module:: pybaram.integrators.dts
.. py:module:: pybaram.integrators.relaxation
.. py:module:: pybaram.solvers
.. py:module:: pybaram.solvers.base.inters
.. py:module:: pybaram.solvers.baseadvec
.. py:module:: pybaram.solvers.baseadvec.inters
.. py:module:: pybaram.solvers.baseadvec.elements
.. py:module:: pybaram.solvers.baseadvec.vertex
.. py:module:: pybaram.solvers.euler.inters
.. py:module:: pybaram.solvers.euler.rsolvers
.. py:module:: pybaram.solvers.navierstokes
.. py:module:: pybaram.solvers.navierstokes.visflux
.. py:module:: pybaram.solvers.rans
.. py:module:: pybaram.solvers.ranssa
.. py:module:: pybaram.solvers.ranssaneg
.. py:module:: pybaram.solvers.ranskwsst
