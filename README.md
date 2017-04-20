## Welcome to Joe Wallwork's MRes project GitHub page!

Here you will find three directories, ``adaptivity``, ``tank_examples`` and
``tsunami``.

The first directory contains test scripts designed to understand and implement
anisotropic mesh adaptivity, as applied to some example PDE problems in
Firedrake.

The second directory contains a Python script called ``tank_master.py``
which enables the user to experiment with the inclusion of non-trivial
bathymetry and a 'wave generator' in a 2D shallow water problem. The domain
considered is a 4m x 1m tank with water depth 10cm. As well as developing a
standalone code to solve this problem, the script makes use of the coastal
and esturarine solver Thetis. This enables the user to generate an accurate
approximation to the true fluid dynamics in the tank, to which the standalone
solution can be compared.

The final directory contains three Python scripts,
``Davis_and_LeVeque_test.py``, ``adjoint_test``and ``tohoku_master.py``.
The 1D Davis and LeVeque tsunami test problem of the
first script comes from Brisa Davis and Randall LeVeque's 2016 paper 'Adjoint
methods for guiding adaptive mesh refinement in tsunami modelling' and studies
the forward and adjoint problems relating to the propagation of an idealised
tsunami across an ocean domain with a step discontinuity (corresponding to a
shelf break). ``adjoint_test`` contains an extension of the code from the 1D
to the 2D case. The 2D shallow water script of ``tohoku_master.py`` builds upon
the standalone forward code developed in ``tank_examples``, along with
``adjoint_test``, and has the application of the 2011 Tohoku tsunami, which
struck the Japanese coast at Fukushima and caused much destruction.
