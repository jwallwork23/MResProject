from firedrake import *
import numpy as np

############################ PARAMETERS ###############################

# Specify problem parameters:
dt = float(raw_input('Specify timestep (default 1): ') or 1.)
Dt = Constant(dt)
n = float(raw_input('Specify number of cells per m (default 5e-4): ') \
        or 5e-4)
T = float(raw_input('Simulation duration in s (default 42000): ') \
          or 4200.)
ndump = int(raw_input('Data dump freq. in timesteps (default 40): ') \
            or 40)

# Set physical and numerical parameters for the scheme:
g = 9.81            # Gravitational acceleration

############################## FE SETUP ###############################

# Define domain and mesh:
lx = 4e5
ly = 1e5
nx = int(lx*n)
ny = int(ly*n)
mesh = RectangleMesh(nx, ny, lx, ly)
x = SpatialCoordinate(mesh)

# Define function spaces:
Vu  = VectorFunctionSpace(mesh, 'CG', 2)    # \ Use Taylor-Hood
Ve = FunctionSpace(mesh, 'CG', 1)           # / elements
Vq = MixedFunctionSpace((Vu, Ve))            

# Construct functions to store forward and adjoint variables:
q_ = Function(Vq)
lam_ = Function(Vq)
u_, eta_ = q_.split()
lu_, le_ = lam_.split()

################# INITIAL CONDITIONS AND BATHYMETRY ###################

# Interpolate ICs:
u_.interpolate(Expression([0, 0]))
eta_.interpolate(Expression('(x[0] >= 1e5) & (x[0] <= 1.5e5) ? \
                             0.4*sin(pi*(x[0]-1e5)*2e-5) : 0.0'))

# Interpolate bathymetry:
b = Function(Ve, name = 'Bathymetry')
b.interpolate(Expression('x[0] <= 50000.0 ? 200.0 : 4000.0'))

###################### FORWARD WEAK PROBLEM ###########################

# Build the weak form of the timestepping algorithm, expressed as a 
# mixed nonlinear problem:
v, ze = TestFunctions(Vq)
q = Function(Vq)
q.assign(q_)
u, eta = split(q)      
u_, eta_ = split(q_)

# Establish forms (functions of the output q), noting we only have a linear
# equation if the stong form is written in terms of a matrix:
L1 = (ze * (eta-eta_) - Dt * inner((eta + b) * u, grad(ze)) + \
      inner(u-u_, v) + Dt * g *(inner(grad(eta), v))) * dx

# Set up the variational problem:
params = {
    'mat_type': 'matfree', 'snes_type': 'ksponly', 'pc_type': 'python',
    'pc_python_type': 'firedrake.AssembledPC',
    'assembled_pc_type': 'lu', 'snes_lag_preconditioner': -1, 
    'snes_lag_preconditioner_persists': True,}
 
uprob1 = NonlinearVariationalProblem(L1, q)
usolver1 = NonlinearVariationalSolver(uprob1, solver_parameters=params)

# Split functions to access their data:
u_, eta_ = q_.split()
u, eta = q.split()

# Store multiple functions:
u.rename('Fluid velocity')
eta.rename('Free surface displacement')

######################## FORWARD TIMESTEPPING #########################

# Initialise files and dump counter:
ufile = File('adjoint_test_outputs/linear_forward.pvd')
t = 0.0
i = 0
dumpn = 0
ufile.write(u, eta, time=t)

# Initialise arrays for storage:
eta_vals = np.zeros((int(T/(ndump*dt))+1, 10251))      # \ TODO: Make these
u_vals = np.zeros((int(T/(ndump*dt))+1, 40501, 2))    # / more general
eta_vals[i,:] = eta.dat.data
u_vals[i,:,:] = u.dat.data

# Enter the forward timeloop:
while (t < T - 0.5*dt):     
    t += dt
    print 't = ', t, ' seconds'
    usolver1.solve()
    q_.assign(q)
    dumpn += 1              # Dump the data
    if (dumpn == ndump):
        dumpn -= ndump
        i += 1
        ufile.write(u, eta, time=t)
        u_vals[i,:,:] = u.dat.data 
        eta_vals[i,:] = eta.dat.data       

print 'Forward problem solved.... now for the adjoint problem.'

################### ADJOINT 'INITIAL' CONDITIONS ######################

# Interpolate ICs:
lu_.interpolate(Expression([0, 0]))
le_.interpolate(Expression('(x[0] >= 1e4) & (x[0] <= 2.5e4) ? \
                            0.4 : 0.0'))

###################### ADJOINT WEAK PROBLEM ###########################

# Establish test functions and split adjoint variables:
w, xi = TestFunctions(Vq)
lam = Function(Vq)
lam.assign(lam_)
lu, le = split(lam)      
lu_, le_ = split(lam_)

# Establish forms (functions of the adjoint output lam):
L2 = (xi * (le-le_) - Dt * g * inner(lu, grad(xi)) + \
      inner(u-u_, w) + Dt * g * inner(grad((eta + b) * le), w)) * dx
                                               # + J derivative term?
# Set up the variational problem
uprob2 = NonlinearVariationalProblem(L2, lam)
usolver2 = NonlinearVariationalSolver(uprob2, solver_parameters=params)

# Split functions in order to access their data:
lu_, le_ = lam_.split()
lu, le = lam.split()

# Store multiple functions:
lu.rename('Adjoint fluid velocity')
le.rename('Adjoint free surface displacement')

######################## BACKWARD TIMESTEPPING ########################

# Initialise dump counter and files:
if (dumpn == 0):
    dumpn = ndump
ufile = File('adjoint_test_outputs/linear_adjoint.pvd')
ufile.write(lu, le, time=t)

# Enter the backward timeloop:
while (t > 0):

    # Update counters:
    t -= dt*ndump               # Longer timestep due to data dumping
    i -= 1
    dumpn -= 1
    print 't = ', t, ' seconds'

    # Solve adjoint problem at current timestep:
    usolver2.solve()
    lam_.assign(lam)

    # Dump data:
    if (dumpn == ndump):
        dumpn -= ndump
        ufile.write(lu, le, time=T-t)   # Note time inversion

    # Assign (forward) free surface values:
    eta.dat.data = eta_vals[i,:]

    # Split adjoint functions:
    lu, le = split(lam)      
    lu_, le_ = split(lam_)

    # Update problem:
    L2 = (xi * (le-le_) + Dt * g * inner(lu, grad(xi)) \
          + inner(lu-lu_, w) + Dt * inner(grad((eta+b) * le), w)) * dx
                                                # + J derivative term?

    # Set up the variational problem
    uprob2 = NonlinearVariationalProblem(L2, lam)
    usolver2 = NonlinearVariationalSolver(uprob2, \
                                          solver_parameters=params)

    # Split functions in order to access their data:
    lu_, le_ = lam_.split()
    lu, le = lam.split()
