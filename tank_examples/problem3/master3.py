from firedrake import *
from thetis import *

########################### USER INPUT ################################

# Specify problem parameters:
mode = raw_input('Use linear or nonlinear equations? (l/n): ') or 'l'
if ((mode != 'l') & (mode != 'n')):
    raise ValueError('Please try again, choosing l or n.')
dt = raw_input('Specify timestep (default 0.01): ') or 0.01
Dt = Constant(dt)
n = raw_input('Specify number of cells per m (default 30): ') or 30
T = raw_input('Specify simulation duration in s (default 40): ') or 40

############################ FE SETUP #################################

# Set physical and numerical parameters for the scheme:
nu = 1e-3           # Viscosity
g = 9.81            # Gravitational acceleration
Cb = 0.0025         # Bottom friction coefficient
depth = 0.1         # Specify tank water depth
ndump = 10
t_export = dt*ndump

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

# Interpolate bathymetry
x = SpatialCoordinate(mesh)
b = Function(Ve, name = 'Bathymetry')
b.interpolate(0.1 + 0.04 * sin(2*pi*x[0]) * sin(2*pi*x[1]))
File('../screenshots/bathymetry.pvd').write(b)

################## INITIAL AND BOUNDARY CONDITIONS ####################

# Interpolate ICs:
u_.interpolate(Expression([0, 0]))
eta_.interpolate(-0.01*cos(0.5*pi*x[0]))

# Apply no-slip BCs on the top and bottom edges of the domain
#bc1 = DirichletBC(W.sub(0), (0.0,0.0), (3,4))

########################## WEAK PROBLEM ###############################

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
    (ze * (eta-eta_) - Dt * inner((eta + b) * u, grad(ze)) + \
    inner(u-u_, v) + Dt * g *(inner(grad(eta), v))) * dx
    )
    return L

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

# The function 'split' has two forms: now use the form which splits a 
# function in order to access its data
u_, eta_ = q_.split()
u, eta = q.split()

############################ TIMESTEPPING #############################

# Store multiple functions
u.rename('Fluid velocity')
eta.rename('Free surface displacement')

# Initialise arrays, files and dump counter
if (mode == 'l'):
    ufile = File('master_output/model_prob3_linear.pvd')
elif (mode == 'n'):
    ufile = File('master_output/model_prob3_nonlinear.pvd')
t = 0.0
ufile.write(u, eta, time=t)
ndump = 10
dumpn = 0

# Create a dictionary containing checkpointed values of eta:
checks ={0.0: eta}

# Enter the timeloop:
while (t < T - 0.5*dt):     
    t += dt
    print 't = ', t, ' seconds'
    usolver.solve()
    q_.assign(q)
    dumpn += 1              # Dump the data
    if dumpn == ndump:
        dumpn -= ndump
        ufile.write(u, eta, time=t)
        # TODO: MAKE THIS MORE GENERAL
        checks[float(int(10*t))/10.0 + 0.1] = eta

print len(checks.keys())    # Sanity check

############################ THETIS SETUP #############################

# Construct solver:
solver_obj = solver2d.FlowSolver2d(mesh, b)
options = solver_obj.options
options.t_export = t_export
options.t_end = T
options.outputdir = 'master_output'

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

##for keys in checks:
    # TO DO

