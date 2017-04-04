from firedrake import *
from thetis import *

################################# USER INPUT ###################################

# Specify problem parameters:
dt = input('Specify timestep (0.01 recommended): ')
Dt = Constant(dt)
n = input('Specify number of mesh cells per m (30 recommended): ')
T = input('Specify simulation duration in s (40 recommended): ')

################################# FE SETUP #####################################

# Set physical and numerical parameters for the scheme:
nu = 1e-3           # Viscosity
g = 9.81            # Gravitational acceleration
Cb = 0.0025         # Bottom friction coefficient
depth = 0.1         # Specify tank water depth
t_export = 0.1  # Export interval in seconds

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
q_ = Function(Vq)            # \ Split means we can interpolate the 
u_, eta_ = q_.split()       # / initial condition into the two components

# Construct a (constant) bathymetry function:
b = Function(Ve, name = 'Bathymetry')
b.assign(depth)

##################### INITIAL AND BOUNDARY CONDITIONS ##########################

# Interpolate ICs:
u_.interpolate(Expression([0, 0]))
eta_.interpolate(Expression('-0.01*cos(0.5*pi*x[0])'))

# Apply no-slip BCs on the top and bottom edges of the domain
#bc1 = DirichletBC(W.sub(0), (0.0,0.0), (3,4))

############################### WEAK PROBLEM ###################################

# Build the weak form of the timestepping algorithm, expressed as a 
# mixed nonlinear problem
v, ze = TestFunctions(Vq)
q = Function(Vq)
q.assign(q_)

# Here we split up a function so it can be inserted into a UFL
# expression
u, eta = split(q)      
u_, eta_ = split(q_)

# Establish form:

def nonlinear_form():
    L = (
        (ze*(eta-eta_) - Dt*inner((eta+b)*u, grad(ze))\
        + inner(u-u_, v) + Dt*(inner(dot(u, nabla_grad(u)), v)\
        + nu*inner(grad(u), grad(v)) + g*inner(grad(eta), v))\
        + Dt*Cb*sqrt(dot(u_, u_))*inner(u/(eta+b), v))*dx(degree=4)
    )
    return L

def linear_form():
    L = (
    (ze*(eta-eta_) - Dt*inner(mu, grad(ze)) + \
    inner(mu-mu_, v) + Dt*g*b*inner(grad(eta), v))*dx   
    )
    return L

# Set up the variational problem

def variational_solver(L, q):
    uprob = NonlinearVariationalProblem(L, q)
    usolver = NonlinearVariationalSolver(uprob,
            solver_parameters={
                            'mat_type': 'matfree',
                            'snes_type': 'ksponly',
                            'pc_type': 'python',
                            'pc_python_type': 'firedrake.AssembledPC',
                            'assembled_pc_type': 'lu',
                    # only rebuild the preconditioner every 10 (-1) solves:
                            'snes_lag_preconditioner': -1, 
                            'snes_lag_preconditioner_persists': True,
                            })
    return usolver

# The function 'split' has two forms: now use the form which splits a 
# function in order to access its data
u_, eta_ = q_.split()
u, eta = q.split()

################################# THETIS SETUP #################################

# Construct solver:
solver_obj = solver2d.FlowSolver2d(mesh, b)
options = solver_obj.options
options.t_export = t_export
options.t_end = T

# Specify integrator of choice:
options.timestepper_type = 'backwardeuler'  # Use implicit timestepping
options.dt = 0.01

# Specify initial surface elevation:
elev_init = Function(P1_2d, name = 'Initial elevation')
x = SpatialCoordinate(mesh)
elev_init.interpolate(-0.01*cos(0.5*pi*x[0]))
solver_obj.assign_initial_conditions(elev=elev_init)

# Run the model:
solver_obj.iterate()

################################# TIMESTEPPING #################################

# Store multiple functions
u.rename('Fluid velocity')
eta.rename('Free surface displacement')

# Initialise arrays, files and dump counter
ufile = File('outputs/error.pvd')
t = 0.0
ufile.write(u, eta, time=t)
ndump = 10
dumpn = 0

# Enter the timeloop:
while (t < T - 0.5*dt):     
    t += dt
    print 't = ', t, ' seconds'
    usolver.solve()
    w_.assign(w)
    dumpn += 1              # Dump the data
    if dumpn == ndump:
        dumpn -= ndump
        ufile.write(u, eta, time=t)
