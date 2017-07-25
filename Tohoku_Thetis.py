from thetis import *

import numpy as np
from math import radians, sin, cos
import scipy.interpolate as si
from scipy.io.netcdf import NetCDFFile

from utils import Tohoku_domain, vectorlonlat2utm


# Define initial mesh (courtesy of QMESH) and functions, with initial conditions set:
try:
    mesh, W, q_, u_, eta_, lam_, lu_, le_, b = Tohoku_domain(int(raw_input('Mesh coarseness? (Integer in 1-5): ') or 4))
except:
    ValueError('Input not recognised. Try entering a natural number less than or equal to 5.')

# Specify time parameters:
dt = float(raw_input('Specify timestep (s) (default 1):') or 1)
ndump = 60      # Inverse data dump frequency
T = 3600        # Simulation time period (s) of 1 hour

# Construct solver:
solver_obj = solver2d.FlowSolver2d(mesh, b)
options = solver_obj.options
options.t_export = dt * ndump
options.t_end = T
options.timestepper_type = 'ssprk33'      # 3-stage, 3rd order Strong Stability Preserving Runge Kutta timestepping
options.dt = dt
options.outputdir = 'plots/tsunami_outputs'

# Apply ICs:
solver_obj.assign_initial_conditions(elev=eta0)

# Run the model:
solver_obj.iterate()
