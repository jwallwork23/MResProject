from thetis import *

# For imposing time-dependent BCs:
def wave_machine(t, A, p, in_flux):
    """Time-dependent flux function"""
    return A * sin(2 * pi * t / p) + in_flux

# Specify tank geometry:
n = 30
lx = 4
ly = 1
nx = lx*n
ny = ly*n
mesh = RectangleMesh(nx, ny, lx, ly)

# Set BC parameters
T = 40.0            # End-time of simulation
A = 0.01            # 'Tide' amplitude
p = 0.5             # 'Tide' period
in_flux = 0         # Flux into domain

# Construct a (constant) bathymetry function:
P1_2d = FunctionSpace(mesh, 'CG', 1)
b = Function(P1_2d, name = 'Bathymetry')
depth = 0.1
b.assign(depth)

# Set end-time parameters:
T = 40.0        # End time in seconds
t_export = 0.1  # Export interval in seconds

# Construct solver:
solver_obj = solver2d.FlowSolver2d(mesh, b)
options = solver_obj.options
options.t_export = t_export
options.t_end = T
options.outputdir = 'prob2_outputs'

# Specify integrator of choice:
options.timestepper_type = 'backwardeuler'      # Use implicit timestepping
options.dt = 0.01

# Define boundary IDs of the domain, for convenience
left_bnd_id = 1
right_bnd_id = 2

# Specify BCs as a dictionary
swe_bnd = {}
in_flux = 0
swe_bnd[right_bnd_id] = {'elev': Constant(0.0),
                         'flux': Constant(-in_flux)}
# NOTE: -ve value => flow into domain. ( Defined as outward normal flux)

# Initialise BCs
tide_flux_const = Constant(wave_machine(0, A, p, in_flux))
swe_bnd[left_bnd_id] = {'flux': tide_flux_const}

# Assign BCs to solver object
solver_obj.bnd_functions['shallow_water'] = swe_bnd
# NOTE: If BCs are not assigned for some boundaries (the lateral boundaries 3
# and 4 in this case), Thetis assumes impermeable land conditions.

# Re-evaluate the BCs as the simulation progresses
def update_forcings(t_new):
    """Callback function that updates all time dependent forcing fields"""
    tide_flux_const.assign(wave_machine(t_new, A, p, in_flux))

# Pass this callback to the time integrator
solver_obj.iterate(update_forcings=update_forcings)
