from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

from utils import *

# Specify problem parameters:
dt = float(raw_input('Specify timestep (default 10): ') or 10.)
Dt = Constant(dt)
n = float(raw_input('Specify number of cells per m (default 5e-4): ') or 5e-4)
T = float(raw_input('Simulation duration in s (default 4200): ') or 4200.)
remesh = raw_input('Use adaptive meshing (y/n)?: ') or 'y'
if remesh == 'y':
    rm = int(raw_input('Timesteps per remesh (default 6)?: ') or 6)     # TODO: consider adaptive remeshing?
else:
    rm = int(T/dt)
    if remesh != 'n':
        raise ValueError('Please try again, typing y or n.')
ntype = raw_input('Normalisation type? (lp/manual): ') or 'lp'
ndump = 1

# Establish problem domain and variables:
mesh, Vq, q_, mu_, eta_, lam_, lm_, le_, b, BCs = tank_domain(n, test2d='y')
nx = int(4e5*n); ny = int(4e5*n)    # TODO: avoid this

# Initialise forward solver:
t = 0.0
dumpn = 0
mn = 0
cnt = 0
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


q_file.write(mu, eta, time=t)

# Enter the forward timeloop:
while t < T - 0.5*dt:

    # Update counters:
    mn += 1
    cnt = 0

    if t != 0:

        # Build Hessian and (hence) metric:
        Vm = TensorFunctionSpace(mesh, 'CG', 1)
        H = construct_hessian(mesh, Vm, eta)
        if remesh == 'y':
            M = compute_steady_metric(mesh, Vm, H, eta, normalise=ntype)
        else:
            M.interpolate(Expression([[n * n, 0], [0, n * n]]))

        # Adapt mesh and update FE setup:
        mesh_ = mesh
        mesh = adapt(mesh, M)
        q_, q, mu_, mu, eta_, eta, b, Vq = update_SW_FE(mesh_, mesh, mu_, mu, eta_, eta, b)

        # Set up functions of weak problem:
        v, ze = TestFunctions(Vq)
        mu, eta = split(q)
        mu_, eta_ = split(q_)

        # Establish form:
        L = linear_form_2d(mu, mu_, eta, eta_, v, ze, b, Dt, n)

        # Set up the variational problem
        q_prob = NonlinearVariationalProblem(L, q)
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

# Initialise adjoint problem:
cnt = 0
mn = 0
lam = Function(Vq)
lam.assign(lam_)
lam_file = File('plots/adjoint_test_outputs/linear_adjoint.pvd')

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

# Initialise dump counter and files:
if dumpn == 0:
    dumpn = ndump

lam_file.write(lm, le, time=0)

# Enter the backward timeloop:
while t > 0:

    # Update counters:
    mn += 1
    cnt = 0

    if t != 0:  # TODO: why is immediate remeshing so slow?

        # Build Hessian and (hence) metric:
        Vm = TensorFunctionSpace(mesh, 'CG', 1)
        H = construct_hessian(mesh, Vm, le)
        if remesh == 'y':
            M = compute_steady_metric(mesh, Vm, H, le, normalise=ntype)
        else:
            M.interpolate(Expression([[n * n, 0], [0, n * n]]))

        # Adapt mesh and update FE setup:
        mesh_ = mesh
        mesh = adapt(mesh, M)
        lam_, lam, lm_, lm, le_, le, b, Vq = update_SW_FE(mesh_, mesh, lm_, lm, le_, le, b)

        # Set up functions of weak problem:
        w, xi = TestFunctions(Vq)
        lm, le = split(q)
        lm_, le_ = split(q_)

        # Establish form:
        L = adj_linear_form_2d(lm, lm_, le, le_, w, xi, b, Dt, n)

        # Set up the variational problem
        lam_prob = NonlinearVariationalProblem(L, lam)
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