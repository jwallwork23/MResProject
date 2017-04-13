## Welcome to Joe Wallwork's MRes project GitHub page

Here you will find two directories, ``tank_examples'' and ``tsunami''.

The former contains a Python script called ``tank_master.py'' which enables
the user to experiment with the inclusion of non-trivial bathymetry and a
``wave generator'' in a 2D shallow water problem in a 4m x 1m tank of water
depth 10cm. As well as developing a standalone code, the script makes use of
the coastal solver Thetis, in order to generate an accurate approximation to
the true dynamics, to which the standalone solution can be compared.

The latter contains two Python scripts, ``Davis_and_LeVeque_test.py'' and
``tohoku_master.py''. The 1D Davis and LeVeque tsunami test problem comes
from their 2016 paper ``Adjoint methods for guiding adaptive mesh refinement
in tsunami modelling'' and studies the forward and adjoint problems relating
to the propagation of an idealised tsunami across an ocean domain with a step
discontinuity (corresponding to a shelf break). The 2D shallow water script
of ``tohoku_master.py'' builds upon the standalone code developed in the tank
example, and has the application of the 2011 Tohoku tsunami, which struck the
Japanese coast at Fukushima and caused much destruction.
