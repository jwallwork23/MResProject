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
Vmu  = VectorFunctionSpace(mesh, 'CG', 2)   # \ Use Taylor-Hood
Ve = FunctionSpace(mesh, 'CG', 1)           # / elements
Vq = MixedFunctionSpace((Vmu, Ve))            

# Construct functions to store forward and adjoint variables:
q_ = Function(Vq)
lam_ = Function(Vq)
mu_, eta_ = q_.split()
lm_, le_ = lam_.split()

################# INITIAL CONDITIONS AND BATHYMETRY ###################

# Interpolate ICs:
mu_.interpolate(Expression([0, 0]))
eta_.interpolate(Expression('(x[0] >= 1e5) & (x[0] <= 1.5e5) ? \
                             0.4*sin(pi*(x[0]-1e5)*2e-5) : 0.0'))

# Interpolate bathymetry:
b = Function(Ve, name = 'Bathymetry')
b.interpolate(Expression('x[0] <= 50000.0 ? 200.0 : 4000.0'))

###################### FORWARD WEAK PROBLEM ###########################

# Build the weak form of the timestepping algorithm, expressed as a 
# mixed nonlinear problem:
nu, ze = TestFunctions(Vq)
q = Function(Vq)
q.assign(q_)
mu, eta = split(q)      
mu_, eta_ = split(q_)

# Establish forms (functions of the output q), noting we only have a 
# linear equation if the stong form is written in terms of a matrix:
L1 = ((eta-eta_) * ze - Dt * inner(mu, grad(ze)) + \
      inner(mu-mu_, nu) + Dt * g * b * (inner(grad(eta), nu))) * dx

# Set up the variational problem:
params = {
    'mat_type': 'matfree', 'snes_type': 'ksponly', 'pc_type': 'python',
    'pc_python_type': 'firedrake.AssembledPC',
    'assembled_pc_type': 'lu', 'snes_lag_preconditioner': -1, 
    'snes_lag_preconditioner_persists': True,}
 
uprob1 = NonlinearVariationalProblem(L1, q)
usolver1 = NonlinearVariationalSolver(uprob1, solver_parameters=params)

# Split functions to access their data:
mu_, eta_ = q_.split()
mu, eta = q.split()

# Store multiple functions:
mu.rename('Fluid momentum')
eta.rename('Free surface displacement')

######################## FORWARD TIMESTEPPING #########################

# Initialise files and dump counter:
ufile1 = File('adjoint_test_outputs/linear_forward.pvd')
t = 0.0
i = 0
dumpn = 0
ufile1.write(mu, eta, time=t)

# Initialise arrays for storage:
eta_vals = np.zeros((int(T/(ndump*dt))+1, 10251))      # \ TODO: Make these
mu_vals = np.zeros((int(T/(ndump*dt))+1, 40501, 2))    # / more general
eta_vals[i,:] = eta.dat.data
mu_vals[i,:,:] = mu.dat.data

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
        ufile1.write(mu, eta, time=t)
        mu_vals[i,:,:] = mu.dat.data 
        eta_vals[i,:] = eta.dat.data       

print 'Forward problem solved.... now for the adjoint problem.'

################### ADJOINT 'INITIAL' CONDITIONS ######################

# Interpolate ICs:
lm_.interpolate(Expression([0, 0]))
le_.interpolate(Expression('(x[0] >= 1e4) & (x[0] <= 2.5e4) ? \
                            0.4 : 0.0'))

###################### ADJOINT WEAK PROBLEM ###########################

# Establish test functions and split adjoint variables:
w, xi = TestFunctions(Vq)
lam = Function(Vq)
lam.assign(lam_)
lm, le = split(lam)      
lm_, le_ = split(lam_)

# Establish forms (functions of the adjoint output lam):
L2 = ((le-le_) * xi - Dt * g * b * inner(lm, grad(xi)) + \
      inner(lm-lm_, w) + Dt * inner(grad(le), w)) * dx
                                               # + J derivative term?
# Set up the variational problem
uprob2 = NonlinearVariationalProblem(L2, lam)
usolver2 = NonlinearVariationalSolver(uprob2, solver_parameters=params)

# Split functions in order to access their data:
lm_, le_ = lam_.split()
lm, le = lam.split()

# Store multiple functions:
lm.rename('Adjoint fluid momentum')
le.rename('Adjoint free surface displacement')

######################## BACKWARD TIMESTEPPING ########################

# Initialise dump counter and files:
if (dumpn == 0):
    dumpn = ndump
ufile2 = File('adjoint_test_outputs/linear_adjoint.pvd')
ufile2.write(lm, le, time=t)

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
        ufile2.write(lm, le, time=T-t)   # Note time inversion
