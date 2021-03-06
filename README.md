## Welcome to Joe Wallwork's MRes project GitHub page! ##

Here you will find:
* A ``utils`` directory, containing the necessary functions for implementation of isotropic and anisotropic mesh
adaptivity:
    * Hessians and metrics can be approximated using ``adaptivity``.
    * Coordinate transformations are achieved using ``conversion``.
    * Meshes, function spaces and functions on a selection of model and ocean domains are generated using ``domain``.
    * Interpolation of functions from an old mesh to a newly adapted mesh is achieved using ``interp``.
    * Time series data can be stored and plotted using ``storage``.
* A ``resources`` directory, containing bathymetry and coastline data for the ocean domain surrounding Fukushima. Mesh
files have been removed for copyright reasons, but may be made available upon request.
* Some test files on model domains:
    * ``1D_tsunami_test``, which replicates results of Brisa Davis and Randall LeVeque's 2016 paper _'Adjoint
    methods for guiding adaptive mesh refinement in tsunami modelling'_ and studies the forward and adjoint problems
    relating to the propagation of an idealised tsunami across an ocean domain with a step discontinuity (corresponding
    to a shelf break).
    * ``sensor_tests``, which tests the adaptive algorithm by adapting the mesh to (stationary) sensor functions as
    used in Geraldine Olivier's 2011 PhD thesis.
    * ``Burgers_test``, which applies anisotropic mesh adaptivity to the case of solving the 2D Burgers' equation,
    with an initial Gaussian profile.
    * ``simple_adaptive_SW``, which applies mesh adaptivity to the shallow water equations in the case of a
    Taylor-Hood __P2__-__P1__ fluid velocity-free surface displacement function space pair.
    * ``goal-based_SW``, which considers a similar problem, but using a goal-based approach to adaptivity.
* Simulations on a realistic domain, which build upon the test script codes and apply the methodology to the 2011 Tohoku
tsunami, which struck the Japanese coast at Fukushima and caused muchdestruction. These include:
    * ``model_verification``, which shows that the linear, non-rotational shallow water equations are sufficient
    and provides an efficient, non-adaptive solver for the tsunami modelling problem.
    * ``simple_adaptive_tsunami``, which solves the problem using anisotropic mesh adaptivity.
    * ``goal-based_tsunami``, which solve the problem using a goal-based approach.
* A ``timeseries`` directory, containing free surface displacement output values at two pressure gauge locations.

For feedback, comments and questions, please email j.wallwork16@imperial.ac.uk.