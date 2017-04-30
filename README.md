## Welcome to Joe Wallwork's MRes project GitHub page!

Here you will find five main script files, ``1D_tsunami_test'',
``2D_tsunami_test'', ``tohoku_master``, ``tank_master`` and ``tank_adaptive''.

The first script test file comes from Brisa Davis and Randall LeVeque's 2016
paper 'Adjoint methods for guiding adaptive mesh refinement in tsunami
modelling' and studies the forward and adjoint problems relating to the
propagation of an idealised tsunami across an ocean domain with a step
discontinuity (corresponding to a shelf break).

The second script test file contains an extension of the code from the 1D
to the 2D case.

The 2D shallow water script of ``tohoku_master.py`` builds upon
the standalone forward code developed in ``tank_examples``, along with
``adjoint_test``, and has the application of the 2011 Tohoku tsunami, which
struck the Japanese coast at Fukushima and caused much destruction.


The 2D shallow water script called ``tank_master.py`` enables the user to
experiment with the inclusion of non-trivial bathymetry and a 'wave generator'
in a 2D shallow water problem. The domain considered is a 4m x 1m tank with
water depth 10cm. As well as developing a standalone code to solve this problem,
the script makes use of the coastal and esturarine solver Thetis. This enables
the user to generate an accurate approximation to the true fluid dynamics in
the tank, to which the standalone solution can be compared.

Finally, the test script ``tank_adaptive'' begins to apply anisotropic mesh
optimisation to the flat bathymetry wave tank problem, with no wave generator.
This will later be incorporated into the master script.
