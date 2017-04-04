from firedrake import *

################################# USER INPUT ###################################

# Specify problem parameters:
dt = input('Specify timestep (0.01 recommended): ')
Dt = Constant(dt)
n = input('Specify number of mesh cells per m (30 recommended): ')
T = input('Specify simulation duration in s (40 recommended): ')

################################# FE SETUP #####################################

# Set physical and numerical parameters for the scheme:
g = 9.81            # Gravitational acceleration
depth = 0.1         # Specify tank water depth
ndump = 10

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

# Construct a function to store our two forward variables at time n:
q_ = Function(Vq)           # \ Split means we can interpolate the 
u_, eta_ = q_.split()       # / initial condition into the two components

# Construct also a function to store the adjoint variables:
lam_ = Function(Vq)
lu_, le_ = lam_.split()

# Construct a (constant) bathymetry function:
b = Function(Ve, name = 'Bathymetry')
b.assign(depth)

############################# FORWARD ICs AND BCs ##############################

# Interpolate ICs:
u_.interpolate(Expression([0, 0]))
eta_.interpolate(Expression('-0.01*cos(0.5*pi*x[0])'))

########################### FORWARD WEAK PROBLEM ###############################

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
L1 = (
    (ze * (eta-eta_) - Dt * inner((eta + b) * u, grad(ze)) + \
    inner(u-u_, v) + Dt * g *(inner(grad(eta), v))) * dx
    )

# Set up the linear problem
uprob = NonlinearVariationalProblem(L1, q)
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

############################# FORWARD TIMESTEPPING #############################

# Store multiple functions
u.rename('Fluid velocity')
eta.rename('Free surface displacement')

# Initialise arrays, files and dump counter
ufile = File('outputs/model_prob1_linear.pvd')
t = 0.0
ufile.write(u, eta, time=t)
dumpn = 0

# Enter the timeloop:
print '='*85
print 'Entering the FORWARD timeloop!'
while (t < T - 0.5*dt):     
    t += dt
    print 't = ', t, ' seconds'
    usolver.solve()
    q_.assign(q)
    dumpn += 1              # Dump the data
    if dumpn == ndump:
        dumpn -= ndump
        ufile.write(u, eta, time=t)

############################# ADJOINT ICs AND BCs ##############################

# Interpolate ICs:
lu_.interpolate(Expression([0, 0]))
le_.interpolate(Expression( \
    '(x[0] < 3.9) && (x[1] > 0.3) && (x[1] < 0.7) ? 1.0 : 0.0'))

########################### ADJOINT WEAK PROBLEM ###############################

# Build the weak form of the timestepping algorithm, expressed as a 
# mixed nonlinear problem
w, xi = TestFunctions(Vq)
lam = Function(Vq)
lam.assign(lam_)

# Here we split up a function so it can be inserted into a UFL
# expression
lu, le = split(lam)      
lu_, le_ = split(lam_)

# NEED TO INCLUDE THE ABILITY TO ACCESS U AND ETA VALUES FROM PVD

# Establish forms (functions of the output w1), noting we only have a linear
# equation if the stong form is written in terms of a matrix:
L2 = (
    (xi * (le-le_) - Dt * g * inner(lu, grad(xi)) + \
    inner(u-u_, w) + Dt * g * inner(grad((eta + b) * le), w)) * dx
    )                                                   # + J derivative term?

# Set up the linear problem
uprob = NonlinearVariationalProblem(L2, lam)
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
lu_, le_ = lam_.split()
lu, le = lam.split()

############################# ADJOINT TIMESTEPPING #############################

# Store multiple functions
lu.rename('Adjoint fluid velocity')
le.rename('Adjoint free surface displacement')

# Initialise arrays, files and dump counter
ufile = File('outputs/model_prob1_linear_adjoint.pvd')
ufile.write(lu, le, time=t)
dumpn = 0

# Enter the timeloop (backwards in time):
print '='*85
print 'Entering the BACKWARD timeloop!'
while (t > 0.5*dt):     
    t -= dt
    print 't = ', T-t, ' seconds'
    usolver.solve()
    lam_.assign(lam)
    dumpn += 1              # Dump the data
    if dumpn == ndump:
        dumpn -= ndump
        ufile.write(lu, le, time=T-t)
