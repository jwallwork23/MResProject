from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

from utils import tank_domain, SW_solve, linear_form_2d, adj_linear_form_2d

# Specify problem parameters:
dt = float(raw_input('Specify timestep (default 10): ') or 10.)
Dt = Constant(dt)
n = float(raw_input('Specify number of cells per m (default 5e-4): ') or 5e-4)
T = float(raw_input('Simulation duration in s (default 4200): ') or 4200.)

# Set physical and numerical parameters for the scheme:
g = 9.81            # Gravitational acceleration
ndump = 5           # Timesteps per data dump

# Initialise mesh and function space:
mesh, Vq, q_, mu_, eta_, lam_, lm_, le_, b, BCs = tank_domain(n, test2d='y')
nx = int(4e5*n)
ny = int(4e5*n)    # TODO: avoid this

# Initialise forward problem:
t = 0.0
i = 0
dumpn = 0
q = Function(Vq)
q.assign(q_)
q_file = File('plots/adjoint_test_outputs/linear_forward.pvd')
params = {'mat_type': 'matfree',
          'snes_type': 'ksponly',
          'pc_type': 'python',
          'pc_python_type': 'firedrake.AssembledPC',
          'assembled_pc_type': 'lu',
          'snes_lag_preconditioner': -1,
          'snes_lag_preconditioner_persists': True,}

# Set up functions of weak problem:
v, ze = TestFunctions(Vq)
mu, eta = split(q)
mu_, eta_ = split(q_)

# Establish form:
L = linear_form_2d(mu, mu_, eta, eta_, v, ze, b, Dt)

# Set up the variational problem
q_prob = NonlinearVariationalProblem(L, q)
q_solv = NonlinearVariationalSolver(q_prob, solver_parameters=params)

# The function 'split' has two forms: now use the form which splits a function in order to access its data
mu_, eta_ = q_.split()
mu, eta = q.split()

mu.rename('Fluid momentum')
eta.rename('Free surface displacement')

# Initialise files and dump counter:
q_file.write(mu, eta, time=t)

# Establish a BC object to get 'coastline'
bc = DirichletBC(Vq.sub(1), 0, 1)
b_nodes = bc.nodes

# Initialise a CG1 version of mu and some arrays for storage:
V1 = VectorFunctionSpace(mesh, 'CG', 1)
mu_cg1 = Function(V1)
mu_cg1.interpolate(mu)
eta_vals = np.zeros((int(T/(ndump*dt))+1, (nx+1)*(ny+1)))
mu_vals = np.zeros((int(T/(ndump*dt))+1, (nx+1)*(ny+1), 2))
m = np.zeros((int(T/(ndump*dt))+1))
eta_vals[i,:] = eta.dat.data
mu_vals[i,:,:] = mu_cg1.dat.data
m[i] = np.log2(max(max(eta_vals[i, b_nodes]), 0.5))

# Enter the forward timeloop:
while t < T - 0.5*dt:
    t += dt
    print 't = ', t, ' seconds'
    q_solv.solve()
    q_.assign(q)
    dumpn += 1
    if dumpn == ndump:
        dumpn -= ndump
        i += 1
        q_file.write(mu, eta, time=t)
        mu_cg1.interpolate(mu)
        mu_vals[i,:,:] = mu_cg1.dat.data
        eta_vals[i,:] = eta.dat.data
        
        # Implement damage measures:
        m[i] = np.log2(max(max(eta_vals[i, b_nodes]), 0.5))

print 'Forward problem solved.... now for the adjoint problem.'

# Initialise adjoint variables:
lam = Function(Vq)
lam.assign(lam_)

# Set up functions of weak problem:
w, xi = TestFunctions(Vq)
lm, le = split(lam)
lm_, le_ = split(lam_)

# Establish form:
L = adj_linear_form_2d(lm, lm_, le, le_, w, xi, b, Dt, n)

# Set up the variational problem
lam_prob = NonlinearVariationalProblem(L, lam)
lam_solv = NonlinearVariationalSolver(lam_prob, solver_parameters=params)

# The function 'split' has two forms: now use the form which splits a function in order to access its data
lm_, le_ = lam_.split()
lm, le = lam.split()

lm.rename('Adjoint fluid momentum')
le.rename('Adjoint free surface displacement')

# Initialise a CG1 version of lm and some arrays for storage:
lm_cg1 = Function(V1)
lm_cg1.interpolate(lm)
le_vals = np.zeros((int(T/(ndump*dt))+1, (nx+1)*(ny+1)))
lm_vals = np.zeros((int(T/(ndump*dt))+1, (nx+1)*(ny+1), 2))
ql_vals = np.zeros((int(T/(ndump*dt))+1, (nx+1)*(ny+1)))
q_dot_lam = Function(Vq.sub(1))
q_dot_lam.rename('Forward-adjoint inner product')
le_vals[i,:] = le.dat.data
lm_vals[i,:] = lm_cg1.dat.data

# Evaluate forward-adjoint inner products (noting mu and lm are in P2, while eta and le are in P1, so we need to evaluate at
# nodes):
ql_vals[i,:] = mu_vals[i,:,0] * lm_vals[i,:,0] + mu_vals[i,:,1] * lm_vals[i,:,1] + eta_vals[i,:] * le_vals[i,:]
q_dot_lam.dat.data[:] = ql_vals[i,:]

# Initialise dump counter and files:
if dumpn == 0:
    dumpn = ndump
lam_file = File('plots/adjoint_test_outputs/linear_adjoint.pvd')
lam_file.write(lm, le, time=0)
dot_file = File('plots/adjoint_test_outputs/inner_product.pvd')
file.write(q_dot_lam, time=0)

# Enter the backward timeloop:
while t > 0:
    t -= dt
    print 't = ', t, ' seconds'
    lam_solv.solve()
    lam_.assign(lam)
    dumpn -= 1
    # Dump data:
    if dumpn == 0:
        dumpn += ndump
        i -= 1
        lm_cg1.interpolate(lm)
        lm_vals[i,:] = lm_cg1.dat.data
        le_vals[i,:] = le.dat.data
        ql_vals[i,:] = mu_vals[i,:,0] * lm_vals[i,:,0] + mu_vals[i,:,1] * lm_vals[i,:,1] + eta_vals[i,:] * le_vals[i,:]
        q_dot_lam.dat.data[:] = ql_vals[i,:]
        # Note the time inversion in outputs:
        lam_file.write(lm, le, time=T-t)
        file.write(q_dot_lam, time=T-t)

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
plt.annotate('Severe damage', xy=(0.7*T, 3), xytext=(0.72*T, 3.2), arrowprops=dict(facecolor='red', shrink=0.05))
plt.annotate('Some inland damage', xy=(0.7*T, 2), xytext=(0.72*T, 2.2), arrowprops=dict(facecolor='orange', shrink=0.05))
plt.annotate('Shore damage', xy=(0.7*T, 1), xytext=(0.72*T, 1.2), arrowprops=dict(facecolor='yellow', shrink=0.05))
plt.annotate('Very little damage', xy=(0.7*T, 0), xytext=(0.72*T, 0.2), arrowprops=dict(facecolor='green', shrink=0.05))
plt.annotate('No damage', xy=(0.7*T, -1), xytext=(0.72*T, -0.8), arrowprops=dict(facecolor='blue', shrink=0.05))
plt.xlabel(r'Time (s)')
plt.ylabel(r'm (dimensionless)')
plt.title(r'Damage measures')
plt.savefig('plots/tsunami_outputs/screenshots/2Ddamage_measures.png')
