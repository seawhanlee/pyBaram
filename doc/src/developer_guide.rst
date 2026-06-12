***************
Developer Guide
***************

Overview of Code Structure
==========================

Start
-----
``pyBaram`` can be executed using the command `pybaram` which is linked to ``__main__.py``. In `run` or `restart` modes, the command calls `process_common` in the :mod:`pybaram.api.simulation` module. Here, the integrator object is initiated, and the run method is called to conduct the simulation.

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
sub-iterations for the block correction. Colored variants use element
coloring to schedule the sweeps by color levels.

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
    :top-classes: pybaram.solver.base.elements.BaseSystem
    :parts: 1 

|

* ``BaseSystem`` : initiates objects and generates kernels from these objects

* ``BaseAdvecSystem`` : `rhside` method for advection problems, such as Euler systems.

    .. admonition:: rhside for advection
      :class: dropdown

        .. automethod:: pybaram.solvers.baseadvec.system.BaseAdvecSystem.rhside

* ``BaseAdvecSystem`` : `rhside` method for advection-diffusion problems, such as Navier-Stokes system.

    .. admonition:: rhside for advection-diffusion
      :class: dropdown

        .. automethod:: pybaram.solvers.baseadvecdiff.system.BaseAdvecDiffSystem.rhside

* ``RANSSystem`` : initiates objects and generates kernels from these objects for RANS simulation


Elements
********
The ``elemenets`` object stores solution and other arrays. It also generates kernels, looping over elements. The class hierarchy can be depicted as follows:

.. inheritance-diagram:: pybaram.solvers.navierstokes.elements
                         pybaram.solvers.euler.elements
    :top-classes: pybaram.solver.base.elements.BaseElements
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
    :top-classes: pybaram.solver.base.elements.BaseElements
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
    :top-classes: pybaram.solver.base.BaseInters
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
    :top-classes: pybaram.solver.base.elements.BaseIntInters
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
    :top-classes: pybaram.solver.base.elements.BaseBCInters
    :parts: 1 

The overall structure and role of these classes are the same as internal interfaces. The  ``construct_bc`` method in ``BaseAdvecInters`` compiles the boundary condition function, and specific formulations are implemented in this class. For example, the hierarchy of boundary conditions for Euler equations can be depicted as follows:

.. inheritance-diagram:: pybaram.solvers.euler.inters.EulerSupOutBCInters
                         pybaram.solvers.euler.inters.EulerSlipWallBCInters
                         pybaram.solvers.euler.inters.EulerSupInBCInters
                         pybaram.solvers.euler.inters.EulerFarInBCInters
                         pybaram.solvers.euler.inters.EulerSubOutPBCInters
    :top-classes: pybaram.solvers.euler.inters.EulerBCInters
    :parts: 1 

The class hierarchy of MPI interfaces can be depicted as follows:

.. inheritance-diagram:: pybaram.solvers.ranskwsst.inters.RANSKWSSTMPIInters
                         pybaram.solvers.ranssa.inters.RANSSAMPIInters
                         pybaram.solvers.ranssaneg.inters.RANSSANegMPIInters
                         pybaram.solvers.navierstokes.inters.NavierStokesMPIInters
                         pybaram.solvers.euler.inters.EulerMPIInters
    :top-classes: pybaram.solver.base.elements.BaseMPIInters
    :parts: 1 

The overall structure and role of these class are the same as internal interfaces.
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
The :mod:`pybaram.backends` module accelerates the pure Python loop and manages the execution of kernels. Currently, only the ``CPUBackend`` is implemented for serial and parallel computation using CPU. This module provides two main features; generating kernels and handling data types for executions.

Kernel Compilation
******************
In the ``integrators`` and the ``solvers`` modules, pure Python functions are defined. These functions are compiled as kernels using loop generators in the :mod:`pybaram.backends.cpu.loops` module. The Numba JIT compiler is then called, and the pure Python functions are compiled to construct serial or parallel loops.

Data Types for Execution
************************
Currently, four data types are defined in the :mod:`pybaram.backends.types`.

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
In :mod:`pybaram.backends.cpu.loop` module, there are two methods: ``make_serial_loop1d`` and ``make_parallel_loop1d``. These methods generate an accelerated kernel from a Python function. A function written in pure Python is compiled using just-in-time compilation with Numba. When ``make_parallel_loop1d`` is used, each thread parallelly executes the loop of this compiled function. Otherwise, the loop of the compiled function is executed sequentially.

.. automodule:: pybaram.backends.cpu.loop
    :members:
    :undoc-members:

As an example, let's consider the ``comm_flux`` function. 
The ``EulerIntInters`` class in the :mod:`pybaram.solvers.euler.inters` module has ``_make_flux`` method, which generates the kernel to compute numerical flux. The ``comm_flux`` function utilizes a plain for loop structure, which is more similar to the loop structure of C/C++ or Fortran than a Pythonic-style one. Therefore, one can readily adopt a well-developed function from a legacy solver into ``pyBaram``. 
The allocation of local arrays was hoisted due to limited functionalities for developing local static variables in Numba. Furthermore, the ``_make_flux`` method passes this Python function to the ``make_serial_loop1d`` or ``make_parallel_loop1d`` method of the backend object and finally returns the serialized or parallelized kernel, respectively.

.. autoclass:: pybaram.solvers.euler.inters.EulerIntInters

  .. method:: _make_flux

The generated kernel is constructed by `construct_kernels` method of ``BaseAdvecIntInters`` 
in :mod:`pybaram.solvers.baseadvec.inters`. When this kernel is called, the reconstructed values at the face
:math:`{U}_f^{\pm}` is used as static argument. 
Thus, ``Kernel`` data type binds this compiled kernel and the static arguments. 
When ``Kernel`` object is called, dynamic arguments can be also provided.
All arguments are parsed, then the compiled kernel is executed.

.. autoclass:: pybaram.solvers.euler.inters.BaseAdvecIntInters

  .. method:: construct_kernels

Non-blocking Send/Receive 
-------------------------
``pyBaram`` exploits the ``mpi4py`` package for MPI communication. Non-blocking communications are employed and overlapped with computing kernels. These methods are implemented in the ``MPIInters`` class.

In the `construct_kernels` method, non-blocking send and receive kernels, along with their requests, are constructed using the `_make_send` and `_make_recv` methods. Buffers are passed to these methods, and the `_sendrecv`` method is invoked. In this method, the `start` function is returned. When this function is called with a `Queue` instance in `rhside`, the MPI request for this communication is registered in the `Queue` instance, and the non-blocking communication starts. This communication is finalized when the `sync` method in the `Queue` instance is called.

.. autoclass:: pybaram.solvers.baseadvec.inters.BaseAdvecMPIInters

    .. method:: construct_kernels

    .. method:: _sendrecv
    
    .. method:: _make_send

    .. method:: _make_recv

.. autoclass:: pybaram.backends.types.Queue
    :noindex:

    .. method:: register

    .. method:: sync
