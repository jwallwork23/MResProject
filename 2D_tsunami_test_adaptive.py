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
    rm = int(raw_input('Timesteps per remesh (default 6)?: ') or 6)     # TODO: consider adaptive remeshing?
    ntype = raw_input('Normalisation type? (lp/manual): ') or 'lp'
else:
    rm = int(T/dt)
    ntype = None
    if remesh != 'n':
        raise ValueError('Please try again, typing y or n.')
ndump = 3       # Timesteps per data dump

# Initialise mesh and function space:
lx = 4e5
nx = int(lx * n)
mesh = SquareMesh(nx, nx, lx, lx)
x = SpatialCoordinate(mesh)

# Define function spaces:
Vu = VectorFunctionSpace(mesh, 'CG', 2)     # \ Taylor-Hood elements
Ve = FunctionSpace(mesh, 'CG', 1)           # /
Vq = MixedFunctionSpace((Vu, Ve))           # Mixed FE problem

# Construct a function to store our two variables at time n:
q_ = Function(Vq)           # Forward solution tuple
u_, eta_ = q_.split()       # Split means we can interpolate the initial condition onto the two components

# Establish bathymetry function:
b = Function(Ve, name = 'Bathymetry')
b.interpolate(Expression('x[0] <= 50000. ? 200. : 4000.'))  # Shelf break bathymetry

# Interpolate initial and boundary conditions, noting higher magnitude wave used due to geometric spreading:
u_.interpolate(Expression([0, 0]))
eta_.interpolate(Expression('(x[0] >= 1e5) & (x[0] <= 1.5e5) & (x[1] >= 1.8e5) & (x[1] <= 2.2e5) ? \
                                        4 * sin(pi*(x[0]-1e5) * 2e-5) * sin(pi*(x[1]-1.8e5) * 2.5e-5) : 0.'))

# Initialise forward solver:
t = 0.0
dumpn = 0
mn = 0
cnt = 0
q = Function(Vq)
q.assign(q_)
q_file = File('plots/adjoint_test_outputs/linear_forward_adaptive.pvd')
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
L1 = linear_form_2d(mu, mu_, eta, eta_, v, ze, b, Dt)

# Set up the variational problem
q_prob = NonlinearVariationalProblem(L1, q)
q_solv = NonlinearVariationalSolver(q_prob, solver_parameters=params)

# The function 'split' has two forms: now use the form which splits a function in order to access its data
mu_, eta_ = q_.split()
mu, eta = q.split()

# Set up outfiles:
mu.rename('Fluid momentum')
eta.rename('Free surface displacement')
q_file.write(mu, eta, time=t)

# Enter the forward timeloop:
while t < T - 0.5*dt:

    # Update counters:
    mn += 1
    cnt = 0

    if (t != 0) & (remesh == 'y'):

        # Build Hessian and (hence) metric:
        Vm = TensorFunctionSpace(mesh, 'CG', 1)
        H = construct_hessian(mesh, Vm, eta)
        M = compute_steady_metric(mesh, Vm, H, eta, normalise=ntype)

        # Adapt mesh and update FE setup:
        mesh_ = mesh
        mesh = adapt(mesh, M)
        q_, q, mu_, mu, eta_, eta, b, Vq = update_SW_FE(mesh_, mesh, mu_, mu, eta_, eta, b)

        # Set up functions of weak problem:
        v, ze = TestFunctions(Vq)
        mu, eta = split(q)
        mu_, eta_ = split(q_)

        # Establish form:
        L1 = linear_form_2d(mu, mu_, eta, eta_, v, ze, b, Dt)

        # Set up the variational problem
        q_prob = NonlinearVariationalProblem(L1, q)
        q_solv = NonlinearVariationalSolver(q_prob, solver_parameters=params)

        # The function 'split' has two forms: now use the form which splits a function in order to access its data
        mu_, eta_ = q_.split()
        mu, eta = q.split()

        # Relabel:
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
            q_file.write(mu, eta, time=t)

print 'Forward problem solved.... now for the adjoint problem.'

# Reset uniform mesh:
#mesh = SquareMesh(nx, nx, lx, lx)

# Re-define function spaces:
Vu = VectorFunctionSpace(mesh, 'CG', 2)     # \ Taylor-Hood elements
Ve = FunctionSpace(mesh, 'CG', 1)           # /
Vq = MixedFunctionSpace((Vu, Ve))           # Mixed FE problem

# Construct a function to store our two variables at time n:
lam_ = Function(Vq)         # Adjoint solution tuple
lu_, le_ = lam_.split()

# Re-establish bathymetry function:
b = Function(Ve, name = 'Bathymetry')
b.interpolate(Expression('x[0] <= 50000. ? 200. : 4000.'))  # Shelf break bathymetry

# Interpolate initial and boundary conditions, noting higher magnitude wave used due to geometric spreading:
lu_.interpolate(Expression([0, 0]))
le_.interpolate(Expression('(x[0] >= 1e4) & (x[0] <= 2.5e4) & (x[1] >= 1.8e5) & (x[1] <= 2.2e5) ? 4 : 0.'))

# Initialise adjoint problem:
cnt = 0
mn = 0
lam = Function(Vq)
lam.assign(lam_)
lam_file = File('plots/adjoint_test_outputs/linear_adjoint_adaptive.pvd')

# Set up functions of weak problem:
w, xi = TestFunctions(Vq)
lm, le = split(lam)
lm_, le_ = split(lam_)

# Establish form:
L2 = adj_linear_form_2d(lm, lm_, le, le_, w, xi, b, Dt)

# Set up the variational problem
lam_prob = NonlinearVariationalProblem(L2, lam)
lam_solv = NonlinearVariationalSolver(lam_prob, solver_parameters=params)

# The function 'split' has two forms: now use the form which splits a function in order to access its data
lm_, le_ = lam_.split()
lm, le = lam.split()

lm.rename('Adjoint fluid momentum')
le.rename('Adjoint free surface displacement')

# Initialise dump counter and files:
if dumpn == 0:
    dumpn = ndump

lam_file.write(lm, le, time=0)

# Enter the backward timeloop:
while t > 0:

    # Update counters:
    mn += 1
    cnt = 0

    if (t != 0) & (remesh == 'y'):  # TODO: why is immediate remeshing so slow?

        # Build Hessian and (hence) metric:
        Vm = TensorFunctionSpace(mesh, 'CG', 1)
        H = construct_hessian(mesh, Vm, le)
        M = compute_steady_metric(mesh, Vm, H, le, normalise=ntype)

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
        lam_solv = NonlinearVariationalSolver(lam_prob, solver_parameters=params)

        # The function 'split' has two forms: now use the form which splits a function in order to access its data
        lm_, le_ = lam_.split()
        lm, le = lam.split()

        # Relabel:
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
        if dumpn == ndump:
            dumpn += ndump
            lam_file.write(lm, le, time=T-t)    # Note the time inversion