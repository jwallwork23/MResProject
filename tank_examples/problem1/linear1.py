from firedrake import *

################################# USER INPUT ###################################

# Specify problem parameters:
dt = raw_input('Specify timestep (default 0.01): ') or 0.01
Dt = Constant(dt)
n = raw_input('Specify number of mesh cells per m (default 30): ') or 30
T = raw_input('Specify simulation duration in s (default 40): ') or 40.0

################################# FE SETUP #####################################

# Set physical and numerical parameters for the scheme:
g = 9.81            # Gravitational acceleration
depth = 0.1         # Specify tank water depth

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

# Establish forms (functions of the output w1), noting we only have a linear
# equation if the stong form is written in terms of a matrix:
L = (
    (ze * (eta-eta_) - Dt * inner((eta + b) * u, grad(ze)) + \
    inner(u-u_, v) + Dt * g *(inner(grad(eta), v))) * dx
    )

# Set up the linear problem
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

# The function 'split' has two forms: now use the form which splits a 
# function in order to access its data
u_, eta_ = q_.split()
u, eta = q.split()

################################# TIMESTEPPING #################################

# Store multiple functions
u.rename('Fluid velocity')
eta.rename('Free surface displacement')

# Initialise arrays, files and dump counter
ufile = File('outputs/model_prob1.pvd')
t = 0.0
ufile.write(u, eta, time=t)
dumpn = 0
checks ={}

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
        checks[t] = eta
