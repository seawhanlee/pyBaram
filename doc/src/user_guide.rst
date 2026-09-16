**********
User Guide
**********

How to Run
==========
pyBaram provides a console script, which uses ``argparse`` module.
When you run ``pybaram``, following help output is given::

    user@Computer ~/pyBaram$ pybaram

    usage: pybaram [-h] [--verbose] {import,partition,run,restart,sweep,export} ...

    positional arguments:
    {import,partition,run,restart,sweep,export}
                            sub-command help
        import              import --help
        partition           partition --help
        run                 run --help
        restart             run --help
        sweep               sweep --help
        export              export --help

    optional arguments:
    -h, --help            show this help message and exit
    --verbose, -v

1. ``pybaram import`` --- Convert mesh-generator output or an existing native
   mesh to a pyBaram mesh file. The output extension selects the rank layout:
   ``.pbrm`` stores a reverse Cuthill-McKee (RCM) rank ordering, while
   ``.pbrmc`` stores a rank coloring required by the colored LU-SGS schemes.
   pyBaram can convert `CGNS <https://cgns.github.io/>`_ mesh (``.cgns``) file or `Gmsh <https://gmsh.info/>`_ mesh file (``.msh``)
    
   Example::

        user@Computer ~/pyBaram$ pybaram import mesh.cgns mesh.pbrm

   You can also scale the mesh by appending the ``-s`` option. For example, to scale by 0.001::

       user@Computer ~/pyBaram$ pybaram import mesh.cgns mesh.pbrm -s 0.001

   For colored output, select ``.pbrmc``. The ``-c`` (or
   ``--coloring-method``) option accepts ``greedy`` (the default) or
   ``smallest-last``::

       user@Computer ~/pyBaram$ pybaram import mesh.cgns mesh.pbrmc -c smallest-last

   Native ``.pbrm`` and ``.pbrmc`` inputs can also be converted between the
   two layouts by choosing the desired output extension.

2. ``pybaram partition`` --- Partition a mesh file for MPI parallel computation.
    
   Example::

        user@Computer ~/pyBaram$ pybaram partition <ranks> mesh.pbrm mesh_p.pbrm

   The partitioned output follows the same extension rule. Use ``.pbrmc`` for
   a colored layout and optionally select its coloring method::

        user@Computer ~/pyBaram$ pybaram partition <ranks> mesh.pbrm mesh_p.pbrmc -c greedy

   You can also partition the solution files associated with a mesh file and save the results to a specified folder::

        user@Computer ~/pyBaram$ pybaram partition <ranks> mesh.pbrm out.pbrs part_folder

3. ``pybaram run`` --- Conduct flow simulation with a given mesh and configuration files (``.ini``).

   Example::
        
        user@Computer ~/pyBaram$ pybaram run mesh.pbrm conf.ini

   The progress display can be selected with ``--ui``. Available modes are
   ``rich`` (default) for inline progress and ``none`` to disable progress::

        user@Computer ~/pyBaram$ pybaram run mesh.pbrm conf.ini --ui rich

   Run, restart, sweep, and Python APIs default to Rich. Only MPI rank 0
   displays progress. Non-terminal output prints the final status (per executed
   AOA for sweeps), without live terminal control sequences. The old ``tui`` and
   ``tqdm`` values are no longer accepted; use ``rich`` or omit the option.

   The CPU backend is used by default. Select the CUDA backend with ``-b cuda``
   (or ``--backend cuda``)::

        user@Computer ~/pyBaram$ pybaram run mesh.pbrm conf.ini -b cuda

   If you would like to conduct MPI parallel computation, please use ``mpirun -n <cores>`` to launch ``pybaram`` script. 
   Note that the mesh file should be partitioned by the same number of cores.

   Example::
        
        user@Computer ~/pyBaram$ mpirun -np <ranks> pybaram run mesh_p.pbrm conf.ini

4. ``pybaram restart`` --- Restart flow simulation with a given mesh and solution files.
   If you would like to restart with different numerical methods, please append the configuration file.

   Example::
        
        user@Computer ~/pyBaram$ pybaram restart mesh.pbrm sol-100.pbrs

   Progress display options are also available for restarted simulations::

        user@Computer ~/pyBaram$ pybaram restart mesh.pbrm sol-100.pbrs --ui rich

   The backend can be selected in the same way as for a fresh run::

        user@Computer ~/pyBaram$ pybaram restart mesh.pbrm sol-100.pbrs -b cuda

5. ``pybaram sweep`` --- Run an angle-of-attack sweep from one mesh and base
   configuration file.

   The sweep command modifies ``[constants] aoa`` for each case. Existing
   configuration expressions that use ``aoa`` for free-stream velocity and
   force directions are re-evaluated for each run.

   Run explicit AOA values::

        user@Computer ~/pyBaram$ pybaram sweep mesh.pbrm conf.ini --aoa 0,2,4

   Run an inclusive range::

        user@Computer ~/pyBaram$ pybaram sweep mesh.pbrm conf.ini --aoa-range -2 6 2

   Sweep output is written to ``sweep-aoa`` by default. Each case gets its own
   directory. Positive AOA values use directory names such as ``aoa1`` and
   ``aoa2``; negative values use names such as ``aoan1`` and ``aoan2``.
   The ``sweep-aoa/sweep.csv`` file summarizes the final row from each
   ``force_*.csv`` file. Use ``--out`` to choose another directory and ``--ui``
   to choose the sweep progress display::

        user@Computer ~/pyBaram$ pybaram sweep mesh.pbrm conf.ini --aoa 0,2,4 --out aoa-study --ui rich

   With ``--ui rich``, the sweep command adds a sweep progress bar above the
   normal solver progress display. It shows the number of completed AOA cases
   and the angle currently running while preserving realtime per-case solver
   status. The right side of the CLI lists each target AOA and its
   latest/final residual so completed cases can be compared while the sweep
   continues.

   Existing non-empty case directories are rejected to avoid appending new
   force and statistics rows to old CSV files. Use ``--overwrite`` only when
   replacing previous sweep results is intended::

        user@Computer ~/pyBaram$ pybaram sweep mesh.pbrm conf.ini --aoa 0,2,4 --out aoa-study --overwrite

   Use ``--resume`` to continue a previous sweep without rerunning non-empty
   case directories. Existing cases are marked as ``skipped`` in
   ``sweep.csv``, new cases are marked as ``complete``, and the summary file is
   updated after each case::

        user@Computer ~/pyBaram$ pybaram sweep mesh.pbrm conf.ini --aoa 0,2,4 --out aoa-study --resume

6. ``pybaram export`` --- Convert solution files to `VTK <https://vtk.org/>`_ unstructured grid file (``.vtu``) 
   or `Tecplot <https://www.tecplot.com/>`_ data file (``.plt``). In addition to volume export, this command can
   export solution data on a specified surface boundary and print the list of available surface names in the mesh.
   The exported volume variables are:

   * All systems: ``rho``, ``p``, and the velocity vector ``uv`` or ``uvw``.
   * Navier-Stokes: the common variables plus ``mu``.
   * Spalart-Allmaras and SA-neg: the common variables plus ``nut``, ``ydist``,
     ``mu``, and ``mut``.
   * All k-omega SST variants: the common variables plus ``k``, ``omega``,
     ``ydist``, ``mu``, and ``mut``.

   Surface export writes ``rho``, ``p``, and the face-normal vector ``n`` for
   all systems. Viscous systems additionally write ``mu`` and the wall shear
   rate vector ``wsr``; RANS systems also write ``ydist`` and ``mut``. The wall
   shear stress vector can be obtained by multiplying ``wsr`` by ``mu``.

   If any selected surface is an ``isotherm-wall``, surface export also writes
   ``WallNormalEnthalpyGradient``. Its value is approximated from the wall
   enthalpy ``CpTw`` on isothermal walls, set to zero on ``adia-wall``
   boundaries, and set to ``NaN`` on other selected boundary types.

   VTK output uses the display names ``Density``, ``Pressure``, ``Velocity``,
   ``Normal``, ``WallShearRate``, and ``WallDistance`` for ``rho``, ``p``, the
   velocity vector, ``n``, ``wsr``, and ``ydist``, respectively. Other variable
   names are written as shown above.

   Example::
        
        user@Computer ~/pyBaram$ pybaram export mesh.pbrm sol-100.pbrs out.vtu

   Export solution data on one or more surface boundaries. For multiple boundaries, use a comma-separated list with ``-s``::

        user@Computer ~/pyBaram$ pybaram export mesh.pbrm sol-100.pbrs surf.vtu -s wall,inlet,outlet

   Print available surface names::

        user@Computer ~/pyBaram$ pybaram export mesh.pbrm --list-surfaces


Mesh File
---------
``pyBaram`` can handle unstructured mixed elements; however, there are some limitations. Currently, only a single unstructured zone can be solved. It is important that volumes and faces are appropriately labeled. The volume label for a single zone should be set as fluid, and faces assigned for boundary conditions must have distinct labels.

Mesh Formats for Implicit Schemes
*********************************
The output extension used by ``import`` or ``partition`` determines the
rank-wide cell layout stored in the native mesh. The layout must match the
implicit scheme selected in the configuration.

.. list-table:: Native mesh format required by each scheme
   :widths: 15 25 60
   :header-rows: 1

   * - Extension
     - Rank layout
     - Compatible schemes
   * - ``.pbrm``
     - RCM rank ordering
     - ``lu-sgs``, ``blu-sgs``, ``petsc``, and ``petsc-rank``
   * - ``.pbrmc``
     - Rank coloring
     - ``colored-lu-sgs`` and ``colored-blu-sgs``
   * - Either format
     - Layout is not used
     - Explicit schemes: ``eulerexplicit``, ``tvd-rk3``, and ``rk5``

For a steady simulation, the ``stepper`` in
``[solver-time-integrator]`` determines the required mesh format. For example,
``stepper = petsc`` requires ``.pbrm``, while
``stepper = colored-blu-sgs`` requires ``.pbrmc``.

For a dual-time stepping simulation, ``bdf1``, ``bdf2``, or ``bdf3`` only
selects the physical-time formula. The mesh format is determined by ``method``
in ``[solver-time-relaxation]``. A DTS simulation using
``method = colored-lu-sgs`` therefore requires ``.pbrmc``; one using
``method = lu-sgs`` or ``method = petsc`` requires ``.pbrm``.

Create the format needed by the selected scheme by choosing the corresponding
output extension::

    user@Computer ~/pyBaram$ pybaram import mesh.cgns mesh.pbrm
    user@Computer ~/pyBaram$ pybaram import mesh.cgns mesh.pbrmc -c smallest-last

The same rule applies to partitioned output::

    user@Computer ~/pyBaram$ pybaram partition <ranks> mesh.pbrm mesh_part.pbrm
    user@Computer ~/pyBaram$ pybaram partition <ranks> mesh.pbrm mesh_part.pbrmc -c greedy

An existing native mesh can be converted to the other layout by changing the
output extension::

    user@Computer ~/pyBaram$ pybaram import mesh.pbrm mesh.pbrmc

If an implicit scheme and mesh layout do not match, pyBaram stops during
solver initialization and reports the layout and extension required by that
scheme. The mesh extension is therefore part of the numerical setup, not only
a file-naming convention.


Configuration File
==================
The parameters for ``pyBaram`` simulation are specified in the configuration file. This file is written in the INI file format, and it is parsed using the ``configparser`` module. The following sections provide details on the sections and parameters.

Backends
---------
The execution backend is selected with ``--backend`` (or ``-b``) on the
``run`` and ``restart`` commands. ``cpu`` is the default, and ``cuda`` selects
the CUDA backend. Backend-specific parameters are configured in the
``[backend-cpu]`` and ``[backend-cuda]`` sections.

Explicit Runge-Kutta schemes support both OpenMP and CUDA execution. Among the
implicit schemes, OpenMP and CUDA are supported only by ``colored-lu-sgs`` and
``colored-blu-sgs``. The non-colored LU-SGS, block LU-SGS, and PETSc schemes
must use the CPU backend with ``multi-thread = single``.

[backend-cpu]
*************
Parameterize CPU backend with

1. multi-thread --- for selecting the multi-threading layer. This parameter passes to ``Numba``.

    ``single`` | ``parallel`` | ``omp`` | ``tbb`` 

    where

        * ``single`` --- use only one thread for the program. This is the default value. 
          If you are running with only MPI parallel computation, please use it. 
          Some numerical schemes only support single thread option.
        
        * ``parallel`` --- use the default multi-threading layer of ``Numba``. 
          Depending on the libraries, ``omp`` or ``tbb`` is used.

        * ``omp`` --- use `OpenMP <https://www.openmp.org/>`_ multi-threading layer. 

        * ``tbb`` --- use `Intel Threading building Blocks <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onetbb.html>`_ multi-threading layer.

Example::

    [backend-cpu]
    multi-thread = parallel

[backend-cuda]
**************
Parameterize the CUDA backend with:

1. threads-per-block --- number of CUDA threads in each kernel block. The
   default value is ``128``.

    `int`

2. cpu-workers --- number of CPU workers used by SciPy when querying the
   wall-distance search tree for RANS simulations. By default, pyBaram divides
   the CPUs available to the process by the number of local MPI ranks and uses
   at least one worker per rank.

    `int`

With MPI, each local rank selects a CUDA device using its local rank. If there
are more local ranks than devices, devices are assigned cyclically.

Example::

    [backend-cuda]
    threads-per-block = 128
    cpu-workers = 4

Constants
---------
In the constants section, essential and user-defined constants are configured. 
Some constants can be expressed as a function of other constants.
The following constants are essential, depending on the equations being solved.

1. gamma --- ratio of the specific heats. For conventional air, :math:`\gamma=1.4`.
   All compressible equations need it.

    `float`

2. mu --- dynamic viscosity. It should be defined when using a constant-viscosity model. This parameter is not required when viscosity is computed using Sutherland's law.

    `float`

3. Pr --- Prandtl number. It should be defined for viscous simulation.
   For conventional air, :math:`Pr=0.72`.

    `float`

4. Prt --- Turbulent Prandtl number. It should be defined for turbulent simulation.
   For conventional air, :math:`Prt=0.9`.

    `float`

Example::

    [constants]
    gamma = 1.4
    Pr = 0.72
    Prt = 0.9
    Re = 6.5e6
    mach = 0.729
    rhof = 1.0
    uf = %(mach)s
    pf = 1/%(gamma)s
    lref = 1.0
    mu = %(mach)s/%(Re)s*%(lref)s
    nutf = 4*%(mu)s/%(rhof)s

Solvers
-------
In following sections, numerical schemes are configured.

[solver]
********
Type of equations and spatial discretization schemes are configured as follows.

1. system --- type of equations. 

    ``euler`` | ``navier-stokes`` | ``rans-sa`` | ``rans-sa-neg`` | ``rans-kwsst`` | ``rans-kwsst-2003m`` | ``rans-kwsst-v2003m``

    * ``rans-<model>`` --- Reynolds-averaged Navier-Stokes equation with turbulence model. 

        * ``rans-sa`` --- one equation Spalart-Allmaras model

        * ``rans-sa-neg`` --- one equation Spalart-Allmaras negative model

        * ``rans-kwsst`` --- alias for ``rans-kwsst-2003m``

        * ``rans-kwsst-2003m`` --- two-equation :math:`k\omega`-SST 2003m
          model using the strain-rate magnitude in the production term

        * ``rans-kwsst-v2003m`` --- v2003m SST variant using the vorticity
          magnitude in the production term

2. order --- spatial order of accuracy.

    ``1`` | ``2``

3. gradient --- method to calculate gradient. The default value is ``hybrid``.

    ``hybrid`` | ``least-square`` | ``weighted-least-square`` | ``green-gauss``

4. hybrid-blend --- blending constant for the ``hybrid`` gradient method.
   The default value is ``2.0``. Smaller values add more Green-Gauss blending
   and may improve robustness on highly anisotropic unstructured meshes.

    `float`

5. pmin --- minimum pressure used when correcting non-physical states. The
   default value is ``1e-15``.

    `float`

5. limiter --- slope limiter for shock-capturing. It is configured only if the order is 2. 
   Default value is ``none``.

    ``none`` | ``mlp-u1`` | ``mlp-u2``

6. u2k --- tuning parameter for MLP-u2 limiter. Normally it is :math:`O(1)`.

    `float`

7. riemann-solver --- scheme to compute inviscid flux at interface.

    ``rusanov`` | ``roe`` | ``roem`` | ``rotated-roem`` | ``hlle`` | ``hllem`` |
    ``ausmpw+`` | ``ausm+up`` | ``ausmzc``

8. viscosity --- method to compute viscosity.
   Default value is ``constant``.

   ``constant`` | ``sutherland``

9. axisymmetric-axis --- axis of symmetry for two-dimensional no-swirl
   axisymmetric simulations. If this option is not specified, the problem is
   treated as a Cartesian two- or three-dimensional simulation.

   ``x`` | ``y``

   For example, ``axisymmetric-axis = x`` means that the computational
   :math:`y` coordinate is the radial coordinate. Euler, Navier-Stokes, and
   RANS systems support this option.

Example::

    [solver]
    system = rans-kwsst
    order = 2
    gradient = hybrid
    hybrid-blend = 2.0
    limiter = mlp-u2
    u2k = 5.0
    riemann-solver = ausmpw+
    viscosity = sutherland
    axisymmetric-axis = x

[solver-source-terms]
************************
Optional source terms for the conservative equations are configured by
conservative-variable name. Unspecified variables use a zero source. Source
expressions may use ``x``, ``y``, and ``z``; values from ``[constants]``; the
conservative variables; and ``sin``, ``cos``, ``exp``, and ``tanh``.

For example, a two-dimensional momentum source can be specified as::

    [solver-source-terms]
    rhou = body_force_x
    rhov = body_force_y

[solver-turbulence-coefficients]
********************************
This optional section overrides turbulence-model constants. Unspecified
coefficients retain their model defaults.

* Spalart-Allmaras: ``cv1 = 7.1``, ``cb1 = 0.1355``, ``cb2 = 0.622``,
  ``sigma = 2/3``, ``kappa = 0.41``, ``cw2 = 0.3``, ``cw3 = 2``,
  ``ct3 = 1.2``, and ``ct4 = 0.5``.
* Negative Spalart-Allmaras additionally uses ``cn1 = 16``.
* k-omega SST: ``sigmak1 = 0.85``, ``sigmaw1 = 0.5``, ``beta1 = 0.075``,
  ``sigmak2 = 1.0``, ``sigmaw2 = 0.856``, ``beta2 = 0.0828``,
  ``betast = 0.09``, ``kappa = 0.41``, ``a1 = 0.31``,
  ``tgamma1 = 5/9``, ``tgamma2 = 0.44``, and ``mut_limit = 1e5``.

[solver-viscosity-sutherland]
*****************************
The parameters associated with Sutherland's law can be configured as follows:

1. muref --- Reference viscosity of the problem. See the `note <https://tmbwg.github.io/turbmodels/Papers/sutherland_notes_cfl3d_fun3d.pdf>`_

    `float`

2. Tref --- Reference temperature of the flow (dimensional quantity).

    `float`

3. CpTf --- Free-stream enthalpy.
    
    `float`

4. Ts --- Sutherland temperature (dimensional quantity). Default value is 110.4 K.

    `float`

5. c1 --- Sutherland constant used to compute the reference viscosity
   (dimensional quantity). The default value corresponds to SI units at
   288.15 K (:math:`1.458\times 10^{-6}`)

    `float`

The quantities **muref** and **CpTf** may be specified in either dimensional or
nondimensional form, depending on the flow variable configuration. The parameters
**Tref** and **Ts** must be given in a consistent dimensional unit system.

If **muref** is provided, it is used directly as the reference viscosity in the
viscosity evaluation.

If **muref** is not provided, it is computed from **c1** and **Tref** using
Sutherland's law as:

.. math::
    \mu_{\infty} = \frac{C_1 T_{\infty}^{3/2}}{T_{\infty} + T_s}

In this case, **CpTf** must be specified in dimensional form consistent with
**Tref**.

Example::

    [solver-viscosity-sutherland]
    muref = rhof*uf*lf/Re
    Tref = 300
    CpTf = gamma / (gamma -1)*pf/rhof
    Ts = 110.4

[solver-time-integrator]
************************
Time integration, relaxation, and dual-time stepping parameters are configured.

1. mode --- type of time integration.

    ``steady`` | ``unsteady`` | ``unsteady-dts``

    * ``steady`` --- pseudo-time iteration to obtain a steady-state solution.

    * ``unsteady`` --- physical-time integration using an explicit scheme.

    * ``unsteady-dts`` --- physical-time integration using dual-time stepping.

2. controller --- method to calculate time step size for ``unsteady`` simulation. 

    ``cfl`` | ``dt``

3. cfl --- Courant - Friedrichs - Lewy Number. 

    `float`

   This parameter is used for ``steady`` simulations and for ``unsteady``
   simulations with ``controller = cfl``.

4. dt --- physical time step size.

    `float`

   This parameter is used for ``unsteady`` simulations with ``controller = dt``
   and for ``unsteady-dts`` simulations.

5. stepper --- method to advance time step.
   For ``unsteady`` simulation, there are following options

    ``eulerexplicit`` | ``tvd-rk3``

   For ``steady`` simulation, following options can be selected.

    ``eulerexplicit`` | ``tvd-rk3`` | ``rk5`` | ``lu-sgs`` | ``colored-lu-sgs`` | ``blu-sgs`` | ``colored-blu-sgs`` | ``petsc`` | ``petsc-rank``

   For ``unsteady-dts`` simulation, following options can be selected.

    ``bdf1`` | ``bdf2`` | ``bdf3``

   In dual-time stepping, higher-order BDF steppers start from lower order
   until enough physical-time history is available: ``bdf2`` starts with BDF1,
   and ``bdf3`` starts with BDF1 and then BDF2.

    * Explicit Runge-Kutta schemes support OpenMP and CUDA. Among the implicit
      schemes, only ``colored-lu-sgs`` and ``colored-blu-sgs`` support OpenMP
      and CUDA.

    * ``lu-sgs``, ``blu-sgs``, ``petsc``, ``petsc-rank`` --- These schemes
      require the CPU backend with the multi-threading layer disabled
      (``multi-thread = single``).

    * ``petsc`` --- One distributed PETSc KSP containing MPI-interface
      couplings. This is the default PETSc path for parallel simulations.

    * ``petsc-rank`` --- An independent ``PETSc.COMM_SELF`` KSP on each MPI
      rank. It retains the rank-local solve path for comparison or tuning.

      Both PETSc methods require ``petsc4py`` and a real, double-precision
      PETSc build.

   Implicit steppers also require a matching native mesh layout. The
   non-colored LU-SGS, block LU-SGS, and PETSc steppers use ``.pbrm``;
   colored LU-SGS and colored block LU-SGS use ``.pbrmc``. See
   `Mesh Formats for Implicit Schemes`_ for the complete mapping.

6. time --- initial and the last physical time for ``unsteady`` and ``unsteady-dts`` simulations.

    `float`, `float`

7. max-iter --- the maximum iteration number for steady simulation

    `int`

8. tolerance --- stopping criteria for the magnitude of residual for steady simulation.

    `float`

9. res-var --- the residual variable to apply tolerance stopping criteria.
   The variable should be selected among the conservative variables. 
   Default variable is ``rho``.

    `string`

10. res-norm --- whether steady residuals are normalized by their initial
    values. The default is ``true``. Set it to ``false`` or ``no`` to use
    absolute residuals for reporting and the ``tolerance`` check.

     `boolean`

11. sub-cfl --- pseudo-time CFL number for ``unsteady-dts`` simulation.

     `float`

12. sub-iter --- The maximum iteration number for sub-iteration process.
    For ``unsteady-dts`` simulation, this is the maximum number of pseudo-time
    sub-iterations per physical time step.

     `int`

13. sub-tol --- The stopping criteria for sub-iteration.
    For ``unsteady-dts`` simulation, this is applied to pseudo-time convergence
    within each physical time step.

     `float`

14. turb-cfl-factor --- The factor of the pseudo-time ``cfl`` number for turbulent equations with respect to that of flow equations.
    It adjusts the pseudo time for turbulence equations to alleviate numerical difficulties. The default value is 1.0.

     `float`

15. visflux-jacobian --- The computing type of viscous Jacobian matrix for several implicit methods.

     ``tlns`` | ``approximate`` | ``none``

    * ``tlns`` --- Based on Thin Layer Navier-Stokes equation (TLNS). Default.

    * ``approximate`` --- Based on Spectral radius. This type computes diagonal elements only.

    * ``none`` --- No viscous flux Jacobian imported. This type can cause convergence delay.

    * Applicable methods --- ``blu-sgs``, ``colored-blu-sgs``, ``petsc``,
      ``petsc-rank``

Example for unsteady simulation::

    [solver-time-integrator]
    controller = cfl
    stepper = tvd-rk3
    time = 0, 0.25
    cfl = 0.9

Example for unsteady simulation with dual-time stepping::

    [solver-time-integrator]
    mode = unsteady-dts
    stepper = bdf2
    time = 0, 0.25
    dt = 1e-3
    sub-cfl = 5.0
    sub-iter = 50
    sub-tol = 1e-3
    res-var = rho

    [solver-time-relaxation]
    method = lu-sgs

Example for steady simulation::

    [solver-time-integrator]
    mode = steady
    cfl = 5.0
    stepper = colored-lu-sgs
    max-iter = 10000
    tolerance = 1e-12
    res-var = rhou

Example for steady simulation with PETSc KSP relaxation::

    [solver-time-integrator]
    mode = steady
    cfl = 5.0
    stepper = petsc
    max-iter = 10000
    tolerance = 1e-12
    res-var = rho

    [solver-petsc]
    ksp = gmres
    preconditioner = ilu
    pc-factor-levels = 0
    sub-iter = 30
    sub-rtol = 1e-3
    sub-atol = 1e-15


[solver-time-relaxation]
************************
Pseudo-time relaxation method for ``unsteady-dts`` simulations is configured.

1. method --- relaxation method for pseudo-time sub-iterations.

    ``lu-sgs`` | ``colored-lu-sgs`` | ``blu-sgs`` | ``colored-blu-sgs`` | ``petsc`` | ``petsc-rank``

   This method determines the mesh format required by a dual-time stepping
   simulation. Non-colored and PETSc methods require ``.pbrm``; colored
   methods require ``.pbrmc``. The outer ``bdf1``, ``bdf2``, or ``bdf3``
   stepper does not change this requirement.

2. sub-iter --- the maximum iteration number for block LU-SGS sub-iteration process.

    `int`

3. sub-rtol --- relative tolerance for block LU-SGS sub-iteration.

    `float`

4. sub-atol --- absolute tolerance for block LU-SGS sub-iteration.

    `float`

[solver-petsc]
**************
PETSc KSP options are configured when ``petsc`` or ``petsc-rank`` is selected
as a steady ``stepper`` or as a dual-time stepping relaxation ``method``.
``petsc`` assembles one distributed BSR operator whose off-rank columns include
MPI-interface cell couplings. ``petsc-rank`` instead creates one independent
rank-local BSR operator per process.

1. ksp --- PETSc Krylov solver type.

    `string`

   The default value is ``gmres``. PETSc's BiCGStab type can be selected with
   ``bcgs``.

2. preconditioner --- PETSc preconditioner type.

    `string`

   The default value is ``ilu``.

   For a parallel global ``petsc`` solve, an ``ilu`` request is implemented
   as zero-overlap additive Schwarz (ASM) with an ILU sub-solver on each rank,
   because PETSc ILU itself is sequential. Other PETSc preconditioner names
   are passed directly to PETSc.

3. pc-factor-levels --- PETSc factor fill level for ILU-like
   preconditioners.

    `int`

   The default value is ``0``.

4. sub-iter --- the maximum number of PETSc KSP iterations.

    `int`

   The default value is ``30``.

5. sub-rtol --- relative tolerance for the PETSc KSP solve.

    `float`

   The default value is ``1e-3``.

6. sub-atol --- absolute tolerance for the PETSc KSP solve.

    `float`

   The default value is ``1e-15``.

7. ksp-divergence --- action when PETSc reports a negative convergence reason.

    ``ignore`` | ``warn`` | ``raise``

   The default value is ``warn``.

Example for dual-time stepping with PETSc KSP relaxation::

    [solver-time-integrator]
    mode = unsteady-dts
    stepper = bdf2
    time = 0, 0.25
    dt = 1e-3
    sub-cfl = 5.0
    sub-iter = 50
    sub-tol = 1e-3

    [solver-time-relaxation]
    method = petsc

    [solver-petsc]
    ksp = bcgs
    preconditioner = ilu
    pc-factor-levels = 0
    sub-iter = 30
    sub-rtol = 1e-3
    sub-atol = 1e-15


[solver-cfl-ramp]
*****************
If this section is configured, CFL number can be ramped up linearly. 
Initially CFL number starts from the assigned ``cfl`` in ``[solver-time-integrator]``.

1. ``iter0`` --- iteration until maintaining the initial CFL.

    `int`

2. ``max-iter`` --- final iteration to finish CFL ramping.

    `int`

3. ``max-cfl`` --- final CFL 

    `float`

Example::

    [solver-cfl-ramp]
    iter0 = 500
    max-iter = 2500
    max-cfl = 10.0

Initial and Boundary Conditions
--------------------------------
Following sections configure initial and boundary conditions. 
The position variables (`x`, `y`, `z`) and 
few numerical functions (:math:`\sin, \cos, \tanh, \exp, \sqrt {}`)
and constant (:math:`\pi`) can be used.

Non-dimensionalization
**********************
``pyBaram`` does not explicitly non-dimensionalize the governing equations. Therefore, it is recommended that users provide appropriately scaled variables for the initial and boundary conditions.

A commonly used nondimensionalization is defined as

.. math::
   \rho^* = \frac{\rho}{\rho_{\infty}}, \quad
   u^* = \frac{u}{a_{\infty}}, \quad
   p^* = \frac{p}{\rho_{\infty} a_{\infty}^2}, \quad
   h^* = \frac{h}{a_{\infty}^2}.

Here, :math:`\rho`, :math:`u`, and :math:`p` denote the density, velocity, and pressure, respectively; :math:`a` denotes the speed of sound, and :math:`h` denotes the specific enthalpy.

For the free-stream state, the nondimensionalized variables become

.. math::
   \rho^*_{\infty} = 1, \quad
   u^*_{\infty} = M_{\infty}, \quad
   p^*_{\infty} = \frac{1}{\gamma}, \quad
   h^*_{\infty} = \frac{1}{\gamma - 1}.

The nondimensional free-stream viscosity :math:`\mu_{\infty}^*` is chosen to satisfy the Reynolds number :math:`Re_L`, defined using a characteristic length
:math:`L`, as

.. math::
   Re_L
   =
   \frac{\rho_{\infty} u_{\infty} L}{\mu_{\infty}}
   =
   \frac{\rho^*_{\infty} u^*_{\infty} L^*}{\mu^*_{\infty}}.

This yields

.. math::
   \mu_{\infty}^*
   =
   \frac{M_{\infty}}{Re_L}\, L^*.

Here, :math:`M_{\infty}` is the free-stream Mach number, and :math:`L^*` is the nondimensional characteristic length (e.g., the chord length used in the mesh).

The viscosity is evaluated using Sutherland's law in nondimensional form as

.. math::
   \mu^*
   =
   \mu_{\infty}^* (T^*)^{3/2}
   \frac{1 + T_s / T_{\infty}}{T^* + T_s / T_{\infty}}.

Here, :math:`T_s` and :math:`T_{\infty}` must be specified in the same dimensional unit system (e.g., Kelvin).


[soln-ics]
**********
The initial condition is configured. All primitive variables should be configured. 

Examples::

    [soln-ics]
    rho = rhof
    u = uf*cos(aoa/180*pi)
    v = uf*sin(aoa/180*pi)
    p = pf

In this example, ``rhof``, ``uf``, ``pf``, and ``aoa`` are assigned in the
``[constants]`` section.

[soln-bcs-`name`]
*****************
The boundary condition for the label ``name`` is configured here. The label
must match a boundary name in the native mesh (``.pbrm`` or ``.pbrmc``).

1. type --- type of boundary condition.
   To solve Euler system, following types can be used.

     ``slip-wall`` | ``sup-out`` | ``sup-in`` | ``sub-outp`` | ``sub-outmdot`` | ``sub-inv`` | ``sub-inptt`` | ``far``

   To solve Navier-Stokes or RANS system, following types can be used.

     ``slip-wall`` | ``adia-wall`` | ``isotherm-wall`` | ``sup-out`` | ``sup-in`` | ``sub-outp`` | ``sub-inv`` | ``sub-inptt`` | ``far``

The details of type and required variables are summarized as follows.

* ``slip-wall`` --- slip wall boundary condition. 

* ``adia-wall`` --- adiabatic wall boundary condition. 

* ``isotherm-wall`` --- isothermal wall boundary condition.

    * ``CpTw`` --- wall enthalpy 

* ``sup-out`` --- supersonic outlet boundary condition

* ``sup-in`` --- supersonic inlet boundary condition

    * `all primitive variables`

* ``sub-outp`` --- subsonic outlet boundary condition with back pressure

    * ``p`` --- back pressure

* ``sub-outmdot`` --- Euler subsonic outlet boundary condition with a specified
  mass flow rate

    * ``mdot`` --- mass flow rate per unit boundary area

    * ``dir`` --- components of the unit vector in the flow direction

* ``sub-inv`` --- subsonic inlet boundary condition with velocity

    * ``rho`` --- density

    * ``u, v, w`` --- velocity components.

    * `turbulent variables`

* ``sub-inptt`` --- subsonic inlet boundary condition with total conditions

    * ``p0`` --- total pressure

    * ``CpT0`` --- total enthalpy

    * ``dir`` --- velocity direction components.

    * `turbulent variables`

* ``far`` --- far boundary condition

    * `all primitive variables`

Examples::

    [soln-bcs-far]
    type = far
    rho = rhof
    u = uf*cos(aoa/180*pi)
    v = uf*sin(aoa/180*pi)
    p = pf

    [soln-bcs-airfoil]
    type = adia-wall

Plugins
--------
Plugins in ``pyBaram`` serve as post-processing modules after iterations. If a plugin is not configured, no post-processing will occur. The following plugins can be configured:

[soln-plugin-stats]
*******************
The `stats` plugin writes a fundamental log file. For unsteady simulations, it includes time and time step information for each iteration. In steady simulations, it records the residuals of all conservative variables.

1. ``flushsteps`` --- flush to the file every ``flushsteps`` iterations. The
   default value is ``500``.

2. ``name`` --- file name. If a file format is not assigned, `csv` format will be used by default. Default name is `stats.csv`

Examples::
    
    [soln-plugin-stats]
    flushsteps = 300


[soln-plugin-writer]
********************
This plugin writes the solution file.

1. ``name`` --- file name. In the name, {n} replaces iteration number and {t} replaces time.

2. ``iter-out`` --- write a solution file every ``iter-out`` iterations for a
   steady simulation. The default value is ``100``.

3. ``dt-out`` --- physical-time interval between solution files for
   ``unsteady`` and ``unsteady-dts`` simulations. This option is required in
   those modes.

Examples::
    
    [soln-plugin-writer]
    name = out-{n}
    iter-out = 5000


[soln-plugin-force-`name`]
**************************
This plugin computes aerodynamic force and moment coefficients along surface labelled `name`.

For axisymmetric simulations, the force and moment are integrated over one
radian in the azimuthal direction. The reference ``area`` should also correspond
to one radian. Multiply the resulting dimensional force or moment by
:math:`2\pi` to obtain the full revolved-surface value.

1. ``iter-out`` --- compute force and moment values every ``iter-out`` iterations for a
   steady simulation.

    `int`

2. ``dt-out`` --- compute force and moment values every ``dt-out`` units of physical
   time for an unsteady simulation.

    `float`

3. ``rho`` --- reference density to compute dynamic pressure

    `float`

4. ``vel`` --- reference velocity to compute dynamic pressure

    `float`

5. ``p`` --- reference pressure which is subtracted from the absolute pressure. 
   The relative pressure is integrated along the surface. The default value is zero.

    `float`

6. ``area`` --- reference area to compute aerodynamic coefficients

    `float`

7. ``length`` --- reference length to compute aerodynamic coefficients

    `float`

8. ``force-dir-name`` --- each character (subscript) denote force direction and its direction will be configured.

    `characters`

9. ``force-dir-`` `character` --- component of force direction vector of each subscript `character`. 
   The dimension of this vector should be the same as the spatial dimension.

    `float`, `float`, ( `float` )

10. ``moment-center`` --- reference position to compute aerodynamic moment.

    `float`, `float`, ( `float` )

11. ``moment-dir-name`` --- each character (subscript) denote moment direction and its direction will be configured.

    `characters`

12. ``moment-dir-`` `character` --- component of moment direction vector of each subscript character. For two-dimensional computation, it is a scalar to indicate whether it is clockwise (-1) or counterclockwise (1). For three-dimensional computation, this vector should have the same dimension as the space.

    `float`, `float`, `float`

Examples::

    [soln-plugin-force-airfoil]
    iter-out = 50
    rho = rhof
    vel = uf
    p = pf
    area = 1.0
    length = 1.0
    force-dir-name = ld
    force-dir-l = -sin(aoa/180*pi), cos(aoa/180*pi)
    force-dir-d = cos(aoa/180*pi), sin(aoa/180*pi)
    moment-center = 0.25, 0
    moment-dir-name = z
    moment-dir-z = -1


[soln-plugin-surface-`name`]
****************************
This plugin integrates variables along the surface labeled as `name`. It provides both integrated and averaged values.

For axisymmetric simulations, the reported integrated value is over one radian
in the azimuthal direction. Multiply by :math:`2\pi` to obtain the full
revolved-surface integral.

1. ``iter-out`` --- compute surface values every ``iter-out`` iterations for a
   steady simulation.

    `int`

2. ``dt-out`` --- compute surface values every ``dt-out`` units of physical
   time for an unsteady simulation.

    `float`

3. ``items`` --- items to integrate. Each item is separated by comma

    `strings`

4. `item` --- expression of `item`. As well as reserved variables for initial and boundary conditions,
   `nx`, `ny`, `nz`, which denote the component normal vector, can be used to express item.

Examples::

    [soln-plugin-surface-pout]
    iter-out = 500
    items = p0, mdot
    p0 = p*(1+ (gamma-1)/2*(u**2 + v**2)/(gamma*p/rho))**(gamma/(gamma-1))
    mdot = rho*(u*nx+v*ny)

In this example, total pressure (:math:`p_0`) and mass flow rate
(:math:`\dot{m}`) are computed.

API
===
pyBaram provides Python APIs for mesh and solution I/O and for running or
restarting simulations. The CLI commands are implemented using these APIs.

.. automodule:: pybaram.api.io
    :members:

.. automodule:: pybaram.api.simulation
    :members:
