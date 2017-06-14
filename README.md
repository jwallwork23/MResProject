## Welcome to Joe Wallwork's MRes project GitHub page!

Here you will find five main script files, ``1D_tsunami_test``,
``adaptive_advection``, ``Gaussian_test``, ``Tohoku_master`` and
``Tohoku_adaptive``. In addition, the directory ``utils`` contains the necessary
functions for implementation of anisotropic mesh adaptivity and the directory
``resources`` contains bathymetry and coastline data. Mesh files have been
removed for copyright reasons, but may be made available upon request.

The first script test file comes from Brisa Davis and Randall LeVeque's 2016
paper 'Adjoint methods for guiding adaptive mesh refinement in tsunami
modelling' and studies the forward and adjoint problems relating to the
propagation of an idealised tsunami across an ocean domain with a step
discontinuity (corresponding to a shelf break).

The second script test file applies anisotropic mesh adaptivity to the case of
a simple 2D advection problem, with an inital Gaussian profile.

The 2D shallow water test script called ``Gaussain_test`` applies mesh
adaptivity in the case of a pair of coupled (vector and scalar, resp.) PDEs, for
P1-P1 elements (Taylor-Hood will be implemented soon).

Finally, the 2D shallow water scripts of ``Tohoku_master`` and
``Tohoku_adaptive`` build upon the test script codes and apply the methodology
to the 2011 Tohoku tsunami, which struck the Japanese coast at Fukushima and
caused much destruction.

For feedback, comments and questions, please email j.wallwork16@imperial.ac.uk.


