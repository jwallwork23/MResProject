from firedrake import *
import matplotlib.pyplot as plt

################################# USER INPUT ###################################

# Specify problem parameters:
dt = input('Specify timestep (1 recommended): ')
Dt = Constant(dt)
n = input('Specify number of mesh cells per m (0.01 recommended): ')
T = input('Specify simulation duration in s (500 recommended): ')
g = 9.81            # Gravitational acceleration

################################### FE SETUP ###################################

# Define domain and mesh
lx = 4e5
nx = lx*n
mesh = IntervalMesh(nx, lx)

# Define function spaces
Vmu = FunctionSpace(mesh, 'CG', 2)    # Use Taylor-Hood elements
Ve = FunctionSpace(mesh, 'CG', 1)           
Vq = MixedFunctionSpace((Vmu, Ve))

# Construct a function to store our two variables at time n
q_ = Function(Vq)        # Split means we can interpolate the 
mu_, eta_ = q_.split()   # initial condition into the two components

# Interpolate bathymetry
b = Function(Ve, name = 'Bathymetry')
b.interpolate(Expression('x[0] <= 50.0 ? 200.0 : 4000.0'))

####################### INITIAL AND BOUNDARY CONDITIONS ########################

# Interpolate ICs
mu_.interpolate(Expression(0))
eta_.interpolate(Expression('exp(-((x[0]-x0)**2)/(2*spread**2))', \
                       x0 = 125.0, spread = 10.0))

################################# WEAK PROBLEM #################################

# Build the weak form of the timestepping algorithm, expressed as a 
# mixed nonlinear problem
v, ze = TestFunctions(Vq)
q = Function(Vq)
q.assign(q_)

# Here we split up a function so it can be inserted into a UFL
# expression
mu, eta = split(q)      
mu_, eta_ = split(q_)

# Establish forms (functions of the output w1), noting we only have a linear
# equation if the stong form is written in terms of a matrix:
L = (
    (ze * (eta-eta_) + Dt * mu.dx(0) *ze + \
    inner(mu-mu_, v) + Dt * g * b* eta.dx(0) * v) * dx
    )

# Set up the problem
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
mu_, eta_ = q_.split()
mu, eta = q.split()

################################# TIMESTEPPING #################################

# Store multiple functions:
mu.rename('Fluid momentum')
eta.rename('Free surface displacement')

# Choose a final time and initialise arrays, files and dump counter:
ufile = File('outputs/Davis_and_LeVeque_test.pvd')
t = 0.0
ufile.write(eta, time=t)
ndump = 10
dumpn = 0
all_us = []

# Enter the timeloop:
while (t < T - 0.5*dt):
    t += dt
    print 't = ', t, ' seconds'
    usolver.solve()
    q_.assign(q)
    dumpn += 1          # Dump the data
    if dumpn == ndump:
        dumpn -= ndump
        ufile.write(eta, time=t)
        all_us.append(Function(eta))

# Plot solution
try:
  plot(all_us)
except Exception as e:
  warning("Cannot plot figure. Error msg: '%s'" % e.message)
try:
  plt.show()
except Exception as e:
  warning("Cannot show figure. Error msg: '%s'" % e.message)
