from firedrake import *
from thetis import *

########################## FORM FUNCTIONS #########################

def nonlinear_form():
    '''Weak residual form of the nonlinear shallow water equations'''
    L = (
     (ze * (eta-eta_) - Dt * inner((eta + b) * u, grad(ze)) + \
      inner(u-u_, v) + Dt * inner(dot(u, nabla_grad(u)), v) + \
      nu * inner(grad(u), grad(v)) + Dt * g * inner(grad(eta), v) + \
      Dt * Cb * sqrt(dot(u_, u_)) * inner(u/(eta+b), v)) * dx(degree=4)   
     )
    return L

def linear_form():
    '''Weak residual form of the linear shallow water equations'''
    L = (
    (ze * (eta-eta_) - Dt * inner((eta + b) * u, grad(ze)) + \
    inner(u-u_, v) + Dt * g *(inner(grad(eta), v))) * dx
    )
    return L

########################### PARAMETERS ################################

# Specify problem parameters:
mode = raw_input('Use linear or nonlinear equations? (l/n): ') or 'l'
if ((mode != 'l') & (mode != 'n')):
    raise ValueError('Please try again, choosing l or n.')
n = raw_input('Specify number of cells per m (default 30): ') or 30
dt = raw_input('Specify timestep (default 0.01): ') or 0.01
Dt = Constant(dt)
ndump = 10
t_export = dt*ndump
T = raw_input('Specify simulation duration in s (default 40): ') or 40

# Set physical parameters for the scheme:
nu = 1e-3           # Viscosity (kg s^{-1} m^{-1})
g = 9.81            # Gravitational acceleration (m s^{-2})
Cb = 0.0025         # Bottom friction coefficient (dimensionless)
depth = 0.1         # Specify tank water depth (m)

############################ FE SETUP #################################

# Define domain and mesh:
lx = 4
ly = 1
nx = lx*n
ny = ly*n
mesh = RectangleMesh(nx, ny, lx, ly)

# Define function spaces:
Vu  = VectorFunctionSpace(mesh, 'CG', 2)    # Use Taylor-Hood elements
Ve = FunctionSpace(mesh, 'CG', 1)           
Vq = MixedFunctionSpace((Vu, Ve))            

# Construct a function to store our two variables at time n:
q_ = Function(Vq)       # \ Split means we can interpolate the 
u_, eta_ = q_.split()   # / initial condition into the two components

# Construct a (constant) bathymetry function:
b = Function(Ve, name = 'Bathymetry')
b.assign(depth)

################## INITIAL AND BOUNDARY CONDITIONS ####################

# Interpolate ICs:
u_.interpolate(Expression([0, 0]))
eta_.interpolate(Expression('-0.01*cos(0.5*pi*x[0])'))

########################## WEAK PROBLEM ###############################

# Build the weak form of the timestepping algorithm, expressed as a 
# mixed nonlinear problem
v, ze = TestFunctions(Vq)
q = Function(Vq)
q.assign(q_)
u, eta = split(q)       # \ Here split means we split up a function so
u_, eta_ = split(q_)    # / it can be inserted into a UFL expression

# Establish form:
if (mode == 'l'):
    L = linear_form()
elif (mode == 'n'):
    L = nonlinear_form()

# Set up the variational problem
uprob = NonlinearVariationalProblem(L, q)
usolver = NonlinearVariationalSolver(uprob,
        solver_parameters={
                            'mat_type': 'matfree',
                            'snes_type': 'ksponly',
                            'pc_type': 'python',
                            'pc_python_type': 'firedrake.AssembledPC',
                            'assembled_pc_type': 'lu',
                            'snes_lag_preconditioner': -1, 
                            'snes_lag_preconditioner_persists': True,
                            })

# Split functions in order to access their data:
u_, eta_ = q_.split()
u, eta = q.split()

# Store multiple functions
u.rename('Fluid velocity')
eta.rename('Free surface displacement')

############################ TIMESTEPPING #############################

# Initialise output directory and dump counter
if (mode == 'l'):
    ufile = File('prob1_outputs/model_prob1_linear.pvd')
elif (mode == 'n'):
    ufile = File('prob1_outputs/model_prob1_nonlinear.pvd')
t = 0.0
dumpn = 0
ufile.write(u, eta, time=t)
eta_sols = [Function(eta)]
u_sols = [Function(u)]

# Enter the timeloop:
while (t < T - 0.5*dt):     
    t += dt
    print 't = ', t, ' seconds'
    usolver.solve()
    q_.assign(q)
    dumpn += 1
    # Dump vtu data:
    if (dumpn == ndump):
        dumpn -= ndump
        ufile.write(u, eta, time=t)
    # Store solution data:
    eta_sols.append(Function(eta))
    u_sols.append(Function(u))

############################ THETIS SETUP #############################

# Construct solver:
solver_obj = solver2d.FlowSolver2d(mesh, b)
options = solver_obj.options
options.t_export = t_export
options.t_end = T
options.outputdir = 'prob1_outputs'

# Specify integrator of choice:
options.timestepper_type = 'backwardeuler'  # Use implicit timestepping
options.dt = dt

# Specify initial surface elevation:
elev_init = Function(Ve, name = 'Initial elevation')
x = SpatialCoordinate(mesh)
elev_init.interpolate(-0.01*cos(0.5*pi*x[0]))
solver_obj.assign_initial_conditions(elev=elev_init)

# Run the model:
solver_obj.iterate()

# OUTPUT CHECKS FOR THETIS TOO

########################### EVALUATE ERROR ############################

# TODO

