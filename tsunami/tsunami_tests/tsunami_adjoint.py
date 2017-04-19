from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

############################ PARAMETERS ###############################

# Specify problem parameters:
dt = float(raw_input('Specify timestep (default 10): ') or 10.)
Dt = Constant(dt)
n = float(raw_input('Specify number of cells per m (default 5e-4): ') \
        or 5e-4)
T = float(raw_input('Simulation duration in s (default 4200): ') \
          or 4200.)

# Set physical and numerical parameters for the scheme:
g = 9.81            # Gravitational acceleration
ndump = 5           # Timesteps per data dump

############################## FE SETUP ###############################

# Define domain and mesh:
lx = 4e5
nx = int(lx*n)
mesh = SquareMesh(nx, nx, lx, lx)
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
eta_.interpolate(Expression('(x[0] >= 1e5) & (x[0] <= 15e4) \
                                & (x[1] >= 18e4) & (x[1] <= 22e4) ? \
                                4 * sin(pi*(x[0]-1e5)/5e4) \
                                * sin(pi*(x[1]-18e4)/4e4) : 0.0'))

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

# Establish a BC object to get 'coastline'
bc = DirichletBC(Ve, 0, 1)
b_nodes = bc.nodes

# Initialise a CG1 version of mu and some arrays for storage:
V1 = VectorFunctionSpace(mesh, 'CG', 1)
mu_cg1 = Function(V1)
mu_cg1.interpolate(mu)
eta_vals = np.zeros((int(T/(ndump*dt))+1, (nx+1)**2))
mu_vals = np.zeros((int(T/(ndump*dt))+1, (nx+1)**2, 2))
m = np.zeros((int(T/(ndump*dt))+1))
eta_vals[i,:] = eta.dat.data
mu_vals[i,:,:] = mu_cg1.dat.data
m[i] = np.log2(max(max(eta_vals[i, b_nodes]), 0.5))

# Enter the forward timeloop:
while (t < T - 0.5*dt):     
    t += dt
    print 't = ', t, ' seconds'
    usolver1.solve()
    q_.assign(q)
    dumpn += 1
    if (dumpn == ndump):
        dumpn -= ndump
        i += 1
        ufile1.write(mu, eta, time=t)
        mu_cg1.interpolate(mu)
        mu_vals[i,:,:] = mu_cg1.dat.data
        eta_vals[i,:] = eta.dat.data
        
        # Implement damage measures:
        m[i] = np.log2(max(max(eta_vals[i, b_nodes]), 0.5))

print 'Forward problem solved.... now for the adjoint problem.'

################### ADJOINT 'INITIAL' CONDITIONS ######################

# Interpolate ICs:
lm_.interpolate(Expression([0, 0]))
le_.interpolate(Expression('(x[0] >= 1e4) & (x[0] <= 2.5e4) \
                            & (x[1] >= 18e4) & (x[1] <= 22e4) ? \
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

# Initialise a CG1 version of lm and some arrays for storage:
lm_cg1 = Function(V1)
lm_cg1.interpolate(lm)
le_vals = np.zeros((int(T/(ndump*dt))+1, (nx+1)**2))
lm_vals = np.zeros((int(T/(ndump*dt))+1, (nx+1)**2, 2))
ql_vals = np.zeros((int(T/(ndump*dt))+1, (nx+1)**2))
q_dot_lam = Function(Ve)
q_dot_lam.rename('Forward-adjoint inner product')
le_vals[i,:] = le.dat.data
lm_vals[i,:] = lm_cg1.dat.data

# Evaluate forward-adjoint inner products (noting mu and lm are in P2,
# while eta and le are in P1, so we need to evaluate at nodes):
ql_vals[i,:] = mu_vals[i,:,0] * lm_vals[i,:,0] + \
                 mu_vals[i,:,1] * lm_vals[i,:,1] + \
                 eta_vals[i,:] * le_vals[i,:]
q_dot_lam.dat.data[:] = ql_vals[i,:]

# Initialise dump counter and files:
if (dumpn == 0):
    dumpn = ndump
ufile2 = File('adjoint_test_outputs/linear_adjoint.pvd')
ufile2.write(lm, le, time=0)
ufile3 = File('adjoint_test_outputs/inner_product.pvd')
ufile3.write(q_dot_lam, time=0)

# Enter the backward timeloop:
while (t > 0):
    t -= dt
    print 't = ', t, ' seconds'
    usolver2.solve()
    lam_.assign(lam)
    dumpn -= 1
    # Dump data:
    if (dumpn == 0):
        dumpn += ndump
        i -= 1
        lm_cg1.interpolate(lm)
        lm_vals[i,:] = lm_cg1.dat.data
        le_vals[i,:] = le.dat.data
        ql_vals[i,:] = mu_vals[i,:,0] * lm_vals[i,:,0] + \
                         mu_vals[i,:,1] * lm_vals[i,:,1] + \
                         eta_vals[i,:] * le_vals[i,:]
        q_dot_lam.dat.data[:] = ql_vals[i,:]
        # Note the time inversion in outputs:
        ufile2.write(lm, le, time=T-t)
        ufile3.write(q_dot_lam, time=T-t)

############################ PLOTTING #################################

# Plot damage measures:
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(range(0, int(T)+1, int(ndump*dt)), m, color='black')
plt.axis([0, T, -1.5, 3.5])
plt.axhline(-1, linestyle='--', color='blue')
plt.axhline(0, linestyle='--', color='green')
plt.axhline(1, linestyle='--', color='yellow')
plt.axhline(2, linestyle='--', color='orange')
plt.axhline(3, linestyle='--', color='red')
plt.annotate('Severe damage', xy=(0.7*T, 3), xytext=(0.72*T, 3.2),
             arrowprops=dict(facecolor='red', shrink=0.05))
plt.annotate('Some inland damage', xy=(0.7*T, 2),
             xytext=(0.72*T, 2.2),
             arrowprops=dict(facecolor='orange', shrink=0.05))
plt.annotate('Shore damage', xy=(0.7*T, 1), xytext=(0.72*T, 1.2),
             arrowprops=dict(facecolor='yellow', shrink=0.05))
plt.annotate('Very little damage', xy=(0.7*T, 0),
             xytext=(0.72*T, 0.2),
             arrowprops=dict(facecolor='green', shrink=0.05))
plt.annotate('No damage', xy=(0.7*T, -1), xytext=(0.72*T, -0.8),
             arrowprops=dict(facecolor='blue', shrink=0.05))
plt.xlabel(r'Time (s)')
plt.ylabel(r'm (dimensionless)')
plt.title(r'Damage measures')
plt.savefig('adjoint_test_outputs/2Ddamage_measures.png')
