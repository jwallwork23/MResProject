from firedrake import *

import numpy as np
from time import clock

from utils import adapt, construct_hessian, compute_steady_metric, tank_domain, update_SW_FE

# Specify problem parameters:
# mode = raw_input('Linear or nonlinear equations? (l/n): ') or 'l'     # TODO: reintroduce nonlinear option
# if (mode != 'l') & (mode != 'n'):
#     raise ValueError('Please try again, choosing l or n.')
bathy = raw_input('Non-trivial bathymetry? (y/n): ') or 'n'
if (bathy != 'y') & (bathy != 'n'):
    raise ValueError('Please try again, choosing y or n.')
dt = float(raw_input('Timestep (default 0.01)?: ') or 0.01)             # TODO: consider adaptive timestepping?
Dt = Constant(dt)
n = int(raw_input('Mesh cells per m (default 16)?: ') or 16)            # Meaning approximately 1000 nodes initially
T = float(raw_input('Simulation duration in s (default 5)?: ') or 5.0)
ndump = int(raw_input('Timesteps per data dump (default 3)') or 3)
remesh = raw_input('Use adaptive meshing (y/n)?: ') or 'y'
if remesh == 'y':
    rm = int(raw_input('Timesteps per remesh (default 6)?: ') or 6)     # TODO: consider adaptive remeshing?
    ntype = raw_input('Normalisation type? (lp/manual): ') or 'lp'
    hmin = float(raw_input('Minimum element size (default 0.005)?: ') or 0.005)
else:
    rm = int(T/dt)
    ntype = None
    if remesh != 'n':
        raise ValueError('Please try again, typing y or n.')
g = 9.81                                                                # Gravitational acceleration (m s^{-2})

# Begin timing:
tic1 = clock()

# Establish tank domain
mesh, Vq, q_, u_, eta_, lam_, lu_, le_, b, BCs = tank_domain(n, bath = bathy)

# Set up functions of the weak problem:
q = Function(Vq)
q.assign(q_)
v, ze = TestFunctions(Vq)
u, eta = split(q)
u_, eta_ = split(q_)

# For timestepping we consider the implicit midpoint rule and so must create new 'mid-step' functions:
uh = 0.5 * (u + u_)
etah = 0.5 * (eta + eta_)

# Specify solver parameters:
params = {'mat_type': 'matfree',
          'snes_type': 'ksponly',
          'pc_type': 'python',
          'pc_python_type': 'firedrake.AssembledPC',
          'assembled_pc_type': 'lu',
          'snes_lag_preconditioner': -1,
          'snes_lag_preconditioner_persists': True,}

# Set up the variational problem:
L = ((eta - eta_) * ze - Dt * (inner(uh * ze, grad(b + etah)) + inner(uh * (b + etah), grad(ze)))
     + inner(u-u_, v) + Dt * g *(inner(grad(etah), v))) * dx
q_prob = NonlinearVariationalProblem(L, q)
q_solv = NonlinearVariationalSolver(q_prob, solver_parameters = params)

# 'Split' functions in order to access their data and then relabel:
u_, eta_ = q_.split()
u, eta = q.split()
u.rename('Fluid velocity')
eta.rename('Free surface displacement')

# Initialisation of counters, functions and files:
t = 0.0
dumpn = 0
mn = 0
q_file = File('plots/prob1_test_outputs/prob1_adapt.pvd')
m_file = File('plots/prob1_test_outputs/prob1_metric.pvd')
q_file.write(u, eta, time = t)

# Enter timeloop:
while t < T-0.5*dt:

    # Update counters:
    mn += 1
    cnt = 0

    if remesh == 'y':

        # Build Hessian and (hence) metric:
        Vm = TensorFunctionSpace(mesh, 'CG', 1)
        H = construct_hessian(mesh, Vm, eta)
        M = compute_steady_metric(mesh, Vm, H, eta, normalise = ntype, h_min = hmin)

        # Adapt mesh and update FE setup:
        mesh_ = mesh
        mesh = adapt(mesh, M)
        q_, q, u_, u, eta_, eta, b, Vq = update_SW_FE(mesh_, mesh, u_, u, eta_, eta, b)

        # Set up functions of weak problem:
        v, ze = TestFunctions(Vq)
        u, eta = split(q)
        u_, eta_ = split(q_)

        # Create 'mid-step' functions:
        uh = 0.5 * (u + u_)
        etah = 0.5 * (eta + eta_)

        # Set up the variational problem
        L = ((eta - eta_) * ze - Dt * (inner(uh * ze, grad(b + etah)) + inner(uh * (b + etah), grad(ze)))
             + inner(u - u_, v) + Dt * g * (inner(grad(etah), v))) * dx
        q_prob = NonlinearVariationalProblem(L, q)
        q_solv = NonlinearVariationalSolver(q_prob, solver_parameters = params)

        # The function 'split' has two forms: now use the form which splits a function in order to access its data
        u_, eta_ = q_.split()
        u, eta = q.split()

        # Relabel and save metric to file:
        u.rename('Fluid velocity')
        eta.rename('Free surface displacement')
        M.rename('Metric field')
        m_file.write(M, time=t)

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
            q_file.write(u, eta, time = t)

# End timing and print:
toc1 = clock()
if remesh == 'y':
    print 'Elapsed time for adaptive tank solver: %1.2es' % (toc1 - tic1)
else:
    print 'Elapsed time for non-adaptive tank solver: %1.2es' % (toc1 - tic1)