## Welcome to Joe Wallwork's MRes project GitHub page! ##

Here you will find:
* A ``utils`` directory, containing the necessary functions for implementation of anisotropic mesh adaptivity.
* A ``resources`` directory, containing bathymetry and coastline data. Mesh files have been removed for copyright
reasons, but may be made available upon request.
* Some test files on model domains:
    * ``1D_tsunami_test``, which replicates results of Brisa Davis and Randall LeVeque's 2016 paper 'Adjoint methods for
    guiding adaptive mesh refinement in tsunami modelling' and studies the forward and adjoint problems relating to the
    propagation of an idealised tsunami across an ocean domain with a step discontinuity (corresponding to a shelf
    break).
    * ``sensor_tests``, which tests the adaptive algorithm by adapting the mesh to (stationary) sensor functions.
    * ``Burgers_test``, which applies anisotropic mesh adaptivity to the case of solving the 2D Burgers' equation, with
    an inital Gaussian profile.
    * ``anisotropic_SW_test``, which applies mesh adaptivity to the shallow water equations in the case of a Taylor-Hood
    P2-P1 fluid velocity-free surface displacement function space pair.
    * ``goal_based_SW_test``, which considers a similar problem, but using a goal-based approach to adaptivity.
* Simulations on a realistic domain, which build upon the test script codes and apply the methodology to the 2011 Tohoku
tsunami, which struck the Japanese coast at Fukushima and caused muchdestruction. These include:
    * ``model_verification``, which shows that the linear, non-rotational shallow water equations are sufficient.
    * ``anisotropic_tsunami``, which solves the problem using anisotropic mesh adaptivity.
    * ``goal_based_tsunami``, which solve the problem using a goal-based approach.

For feedback, comments and questions, please email j.wallwork16@imperial.ac.uk.