from firedrake import *
import matplotlib.pyplot as plt

################################# USER INPUT ###################################

# Specify problem parameters:
dt = raw_input('Specify timestep (default 1): ') or 1
Dt = Constant(dt)
n = raw_input('Specify number of mesh cells per m (default 0.001): ') or 0.001
T = raw_input('Specify simulation duration in s (default 500): ') or 500
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
b.interpolate(Expression('x[0] <= 50000.0 ? 200.0 : 4000.0'))
plot(b)
plt.title('Ocean depth (m)')
plt.show()

####################### INITIAL AND BOUNDARY CONDITIONS ########################

# Interpolate ICs
mu_.interpolate(Expression(0))
x0 = 125000.0
spread = 1000.0
eta_ = Function(Ve)
eta_.interpolate(Expression( \
    '(x[0] > 120000.0) && (x[0] < 130000.0) ? 0.4 : 0.0'))
plot(eta_)
plt.title('Initial surface profile (m)')
plt.show()

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
    (ze * (eta-eta_) - Dt * mu * ze.dx(0) + \
    inner(mu-mu_, v) + Dt * g * b* eta.dx(0) * v) * dx
    )

# Set up the problem
uprob = NonlinearVariationalProblem(L, q)
usolver = NonlinearVariationalSolver(uprob, solver_parameters=
                                     {'mat_type' : 'aij',
                                      'ksp_type' : 'preonly',
                                      'pc_type' : 'lu'})

# The function 'split' has two forms: now use the form which splits a 
# function in order to access its data
mu_, eta_ = q_.split()
mu, eta = q.split()

################################# TIMESTEPPING #################################

# Store multiple functions:
mu.rename('Fluid momentum')
eta.rename('Free surface displacement')

# Choose a final time and initialise arrays, files and dump counter:
##ufile = File('outputs/Davis_and_LeVeque_test.pvd')
t = 0.0
##ufile.write(eta, time=t)
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
        ##ufile.write(eta, time=t)
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
