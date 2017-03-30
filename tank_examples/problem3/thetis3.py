from thetis import *

# Specify tank geometry:
n = 30
lx = 4
ly = 1
nx = lx*n
ny = ly*n
mesh = RectangleMesh(nx, ny, lx, ly)

# Construct a (nontrivial) bathymetry function:
x = SpatialCoordinate(mesh)
P1_2d = FunctionSpace(mesh, 'CG', 1)
b = Function(P1_2d, name = 'Bathymetry')
b.interpolate(0.1 + 0.04 * sin(2*pi*x[0]) * sin(2*pi*x[1]))

# Set end-time parameters:
T = 40.0        # End time in seconds
t_export = 0.1  # Export interval in seconds

# Construct solver:
solver_obj = solver2d.FlowSolver2d(mesh, b)
options = solver_obj.options
options.t_export = t_export
options.t_end = T

# Specify integrator of choice:
options.timestepper_type = 'backwardeuler'      # Use implicit timestepping
options.dt = 0.01

# Specify initial surface elevation:
elev_init = Function(P1_2d, name = 'Initial elevation')
elev_init.interpolate(-0.01*cos(0.5*pi*x[0]))
solver_obj.assign_initial_conditions(elev=elev_init)

# Run the model:
solver_obj.iterate()
