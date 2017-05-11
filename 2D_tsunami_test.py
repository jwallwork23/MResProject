from firedrake import *

import numpy as np
import matplotlib.pyplot as plt

from utils import linear_form_2d, adj_linear_form_2d, construct_hessian, compute_steady_metric, update_SW_FE, adapt

# Specify problem parameters:
dt = float(raw_input('Specify timestep (default 10): ') or 10.)
Dt = Constant(dt)
n = float(raw_input('Specify number of cells per m (default 1e-4): ') or 1e-4)
T = float(raw_input('Simulation duration in s (default 4200): ') or 4200.)
remesh = raw_input('Use adaptive meshing (y/n)?: ') or 'y'
if remesh == 'y':
    rm = int(raw_input('Timesteps per remesh (default 6)?: ') or 6)         # TODO: consider adaptive remeshing?
    ntype = raw_input('Normalisation type? (lp/manual): ') or 'lp'
else:
    rm = int(T/dt)
    ntype = None
    if remesh != 'n':
        raise ValueError('Please try again, typing y or n.')
ndump = 3                                                                   # Timesteps per data dump



# Initialise mesh and function space:
lx = 4e5
nx = int(lx * n)
mesh = SquareMesh(nx, nx, lx, lx)
x = SpatialCoordinate(mesh)

# Define function spaces:
Vu = VectorFunctionSpace(mesh, 'CG', 2)                                     # \ Taylor-Hood elements
Ve = FunctionSpace(mesh, 'CG', 1)                                           # /
Vq = MixedFunctionSpace((Vu, Ve))                                           # Mixed FE problem

# Construct a function to store our two variables at time n:
q_ = Function(Vq)                                                           # Forward solution tuple
u_, eta_ = q_.split()

# Establish bathymetry function:
b = Function(Ve, name = 'Bathymetry')
b.interpolate(Expression('x[0] <= 50000. ? 200. : 4000.'))  # Shelf break bathymetry

# Interpolate initial and boundary conditions, noting higher magnitude wave used due to geometric spreading:
u_.interpolate(Expression([0, 0]))
eta_.interpolate(Expression('(x[0] >= 1e5) & (x[0] <= 1.5e5) & (x[1] >= 1.8e5) & (x[1] <= 2.2e5) ? \
                                        10 * sin(pi*(x[0]-1e5) * 2e-5) * sin(pi*(x[1]-1.8e5) * 2.5e-5) : 0.'))

# Set up functions of forward weak problem:
v, ze = TestFunctions(Vq)
mu, eta = split(q)
mu_, eta_ = split(q_)
q = Function(Vq)
q.assign(q_)

# Specify solver parameters:
params = {'mat_type': 'matfree',
          'snes_type': 'ksponly',
          'pc_type': 'python',
          'pc_python_type': 'firedrake.AssembledPC',
          'assembled_pc_type': 'lu',
          'snes_lag_preconditioner': -1,
          'snes_lag_preconditioner_persists': True,}

# Set up the variational problem
L1 = linear_form_2d(mu, mu_, eta, eta_, v, ze, b, Dt)
q_prob = NonlinearVariationalProblem(L1, q)
q_solv = NonlinearVariationalSolver(q_prob, solver_parameters = params)

# 'Split' functions to access their data and relabel:
mu_, eta_ = q_.split()
mu, eta = q.split()
mu.rename('Fluid momentum')
eta.rename('Free surface displacement')

# Initialise time, counters and files:
t = 0.0
dumpn = 0
mn = 0
cnt = 0
i = 0
q_file = File('plots/adjoint_test_outputs/linear_forward.pvd')
q_file.write(mu, eta, time = t)

if remesh == 'n':

    # Establish a BC object to get 'coastline'
    bc = DirichletBC(Vq.sub(1), 0, 1)
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
while t < T - 0.5*dt:

    # Update counters:
    mn += 1
    cnt = 0

    if (t != 0) & (remesh == 'y'):

        # Build Hessian and (hence) metric:
        Vm = TensorFunctionSpace(mesh, 'CG', 1)
        H = construct_hessian(mesh, Vm, eta)
        M = compute_steady_metric(mesh, Vm, H, eta, normalise = ntype)

        # Adapt mesh and update FE setup:
        mesh_ = mesh
        mesh = adapt(mesh, M)
        q_, q, mu_, mu, eta_, eta, b, Vq = update_SW_FE(mesh_, mesh, mu_, mu, eta_, eta, b)

        # Set up functions of weak problem:
        v, ze = TestFunctions(Vq)
        mu, eta = split(q)
        mu_, eta_ = split(q_)

        # Set up the variational problem:
        L1 = linear_form_2d(mu, mu_, eta, eta_, v, ze, b, Dt)
        q_prob = NonlinearVariationalProblem(L1, q)
        q_solv = NonlinearVariationalSolver(q_prob, solver_parameters = params)

        # 'Split' functions to access their data and relabel:
        mu_, eta_ = q_.split()
        mu, eta = q.split()
        mu.rename('Fluid momentum')
        eta.rename('Free surface displacement')

        # Enter the inner timeloop:
    while cnt < rm:
        t += dt
        print 't = ', t, ' seconds, mesh number = ', mn
        cnt += 1
        q_solv.solve()
        q_.assign(q)
        dumpn += 1
        if dumpn == ndump:
            dumpn -= ndump
            q_file.write(mu, eta, time = t)

            if remesh == 'n':
                i += 1
                mu_cg1.interpolate(mu)
                mu_vals[i, :, :] = mu_cg1.dat.data
                eta_vals[i, :] = eta.dat.data

                # Implement damage measures:                        # TODO: change this as it is dominated by shoaling
                m[i] = np.log2(max(max(eta_vals[i, b_nodes]), 0.5))

print 'Forward problem solved.... now for the adjoint problem.'

if remesh == 'y':

    # Reset uniform mesh:
    mesh = SquareMesh(nx, nx, lx, lx)

    # Re-define function spaces:
    Vu = VectorFunctionSpace(mesh, 'CG', 2)                         # \ Taylor-Hood elements
    Ve = FunctionSpace(mesh, 'CG', 1)                               # /
    Vq = MixedFunctionSpace((Vu, Ve))                               # Mixed FE problem

    # Re-establish bathymetry function:
    b = Function(Ve, name = 'Bathymetry')
    b.interpolate(Expression('x[0] <= 50000. ? 200. : 4000.'))      # Shelf break bathymetry

# Construct a function to store our two variables at time n:
lam_ = Function(Vq)                                                 # Adjoint solution tuple
lu_, le_ = lam_.split()

# Interpolate initial and boundary conditions, noting higher magnitude wave used due to geometric spreading:
lu_.interpolate(Expression([0, 0]))
le_.interpolate(Expression('(x[0] >= 1e4) & (x[0] <= 2.5e4) & (x[1] >= 1.8e5) & (x[1] <= 2.2e5) ? 10 : 0.'))

# Set up functions of weak problem:
lam = Function(Vq)
lam.assign(lam_)
w, xi = TestFunctions(Vq)
lm, le = split(lam)
lm_, le_ = split(lam_)

# Set up the variational problem:
L2 = adj_linear_form_2d(lm, lm_, le, le_, w, xi, b, Dt)
lam_prob = NonlinearVariationalProblem(L2, lam)
lam_solv = NonlinearVariationalSolver(lam_prob, solver_parameters = params)

# 'Split' functions to access their data and relabel:
lm_, le_ = lam_.split()
lm, le = lam.split()
lm.rename('Adjoint fluid momentum')
le.rename('Adjoint free surface displacement')

# Initialise counters and files:
cnt = 0
mn = 0
lam_file = File('plots/adjoint_test_outputs/linear_adjoint.pvd')
lam_file.write(lm, le, time=0)

if remesh == 'n':

    # Initialise a CG1 version of lm and some arrays for storage (with dimension pre-allocated for speed):
    lm_cg1 = Function(V1)
    lm_cg1.interpolate(lm)
    le_vals = np.zeros((int(T/(ndump*dt))+1, (nx+1)**2))
    lm_vals = np.zeros((int(T/(ndump*dt))+1, (nx+1)**2, 2))
    ql_vals = np.zeros((int(T/(ndump*dt))+1, (nx+1)**2))
    q_dot_lam = Function(Vq.sub(1))
    q_dot_lam.rename('Forward-adjoint inner product')
    le_vals[i,:] = le.dat.data
    lm_vals[i,:] = lm_cg1.dat.data

    # Initialise file:
    dot_file = File('plots/adjoint_test_outputs/inner_product.pvd')
    dot_file.write(q_dot_lam, time = 0)

    # Evaluate forward-adjoint inner products (noting mu and lm are in P2, while eta and le are in P1, so we need to
    # evaluate at nodes):
    ql_vals[i,:] = mu_vals[i,:,0] * lm_vals[i,:,0] + mu_vals[i,:,1] * lm_vals[i,:,1] + eta_vals[i,:] * le_vals[i,:]
    q_dot_lam.dat.data[:] = ql_vals[i,:]

# Initialise dump counter:
if dumpn == 0:
    dumpn = ndump

# Enter the backward timeloop:
while t > 0:

    # Update counters:
    mn += 1
    cnt = 0

    if (t != 0) & (remesh == 'y'):                                          # TODO: why is immediate remeshing so slow?

        # Build Hessian and (hence) metric:
        Vm = TensorFunctionSpace(mesh, 'CG', 1)
        H = construct_hessian(mesh, Vm, le)
        M = compute_steady_metric(mesh, Vm, H, le, normalise = ntype)

        # Adapt mesh and update FE setup:
        mesh_ = mesh
        mesh = adapt(mesh, M)
        lam_, lam, lm_, lm, le_, le, b, Vq = update_SW_FE(mesh_, mesh, lm_, lm, le_, le, b)

        # Set up functions of weak problem:
        w, xi = TestFunctions(Vq)
        lm, le = split(q)
        lm_, le_ = split(q_)

        # Establish form:
        L2 = adj_linear_form_2d(lm, lm_, le, le_, w, xi, b, Dt)

        # Set up the variational problem
        lam_prob = NonlinearVariationalProblem(L2, lam)
        lam_solv = NonlinearVariationalSolver(lam_prob, solver_parameters = params)

        # 'Split' functions to access their data and relabel:
        lm_, le_ = lam_.split()
        lm, le = lam.split()
        lm.rename('Fluid momentum')
        le.rename('Free surface displacement')

    # Enter the inner timeloop:
    while cnt < rm:
        t -= dt
        print 't = ', t, ' seconds, mesh number = ', mn
        cnt += 1
        lam_solv.solve()
        lam_.assign(lam)
        dumpn -= 1
        if dumpn == 0:
            dumpn += ndump
            lam_file.write(lm, le, time = T-t)                                        # Note time inversion

            if remesh == 'n':
                i -= 1
                lm_cg1.interpolate(lm)
                lm_vals[i, :] = lm_cg1.dat.data
                le_vals[i, :] = le.dat.data
                ql_vals[i, :] = mu_vals[i, :, 0] * lm_vals[i, :, 0] + mu_vals[i, :, 1] * lm_vals[i, :, 1] + \
                                eta_vals[i, :] * le_vals[i, :]
                q_dot_lam.dat.data[:] = ql_vals[i, :]
                dot_file.write(q_dot_lam, time = t)


# Plot damage measures:
if remesh == 'n':
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
    plt.annotate('Some inland damage', xy=(0.7*T, 2), xytext=(0.72*T, 2.2),
                 arrowprops=dict(facecolor='orange', shrink=0.05))
    plt.annotate('Shore damage', xy=(0.7*T, 1), xytext=(0.72*T, 1.2),
                 arrowprops=dict(facecolor='yellow', shrink=0.05))
    plt.annotate('Very little damage', xy=(0.7*T, 0), xytext=(0.72*T, 0.2),
                 arrowprops=dict(facecolor='green', shrink=0.05))
    plt.annotate('No damage', xy=(0.7*T, -1), xytext=(0.72*T, -0.8), arrowprops=dict(facecolor='blue', shrink=0.05))
    plt.xlabel(r'Time (s)')
    plt.ylabel(r'm (dimensionless)')
    plt.title(r'Damage measures')
    plt.savefig('plots/tsunami_outputs/screenshots/2Ddamage_measures.png')