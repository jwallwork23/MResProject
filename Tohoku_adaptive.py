from firedrake import *

import numpy as np
from time import clock

from utils import adapt, construct_hessian, compute_steady_metric, interp, Tohoku_domain, update_SW_FE

# Define initial mesh (courtesy of QMESH) and functions, with initial conditions set:
res = raw_input('Mesh type fine, medium or coarse? (f/m/c): ') or 'c'
if (res != 'f') & (res != 'm') & (res != 'c'):
    raise ValueError('Please try again, choosing f, m or c.')
mesh, Vq, q_, u_, eta_, lam_, lm_, le_, b = Tohoku_domain(res)

# Choose linear or nonlinear equations:
# mode = raw_input('Linear or nonlinear equations? (l/n): ') or 'l'     # TODO: reintroduce nonlinear option
# if (mode != 'l') & (mode != 'n'):
#     raise ValueError('Please try again, choosing l or n.')

# Specify timestepping parameters:
dt = float(raw_input('Timestep (default 15)?: ') or 15)                 # TODO: consider adaptive timestepping?
Dt = Constant(dt)
T = float(raw_input('Simulation duration in hours (default 2)?: ') or 2.) * 3600.
ndump = 1

# Set up adaptivity parameters:
remesh = raw_input('Use adaptive meshing (y/n)?: ') or 'y'
if remesh == 'y':
    hmin = float(raw_input('Minimum element size (default 0.005)?: ') or 0.005)
    rm = int(raw_input('Timesteps per remesh (default 5)?: ') or 5)
    nodes = float(raw_input('Target number of nodes (default 1000)?: ') or 1000.)
    ntype = raw_input('Normalisation type? (lp/manual): ') or 'lp'
    print 'Initial number of nodes : ', len(mesh.coordinates.dat.data)
else:
    hmin = 0
    rm = int(T/dt)
    nodes = 0
    ntype = None

# Begin timing:
tic1 = clock()

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
g = 9.81
L = ((eta - eta_) * ze - Dt * (inner(uh * ze, grad(b + etah)) + inner(uh * (b + etah), grad(ze)))
     + inner(u-u_, v) + Dt * g *(inner(grad(etah), v))) * dx
q_prob = NonlinearVariationalProblem(L, q)
q_solv = NonlinearVariationalSolver(q_prob, solver_parameters = params)

# 'Split' functions in order to access their data and then relabel:
u_, eta_ = q_.split()
u, eta = q.split()
u.rename('Fluid velocity')
eta.rename('Free surface displacement')

# Initialise counters and files:
t = 0.0
mn = 0
dumpn = 0
q_file = File('plots/adapt_plots/tohoku_adapt.pvd')
m_file = File('plots/adapt_plots/tohoku_adapt_metric.pvd')
q_file.write(u, eta, time = t)

while t < T - 0.5 * dt:

    mn += 1
    cnt = 0

    if remesh == 'y':
        print '************ Adaption step %d **************' % mn

        # Compute Hessian and metric:
        V = TensorFunctionSpace(mesh, 'CG', 1)
        H = construct_hessian(mesh, V, eta)
        M = compute_steady_metric(mesh, V, H, eta, h_min = hmin, N = nodes)
        M.rename('Metric field')
        m_file.write(M, time = t)

        # Adapt mesh and update FE setup:
        mesh_ = mesh
        tic2 = clock()
        mesh = adapt(mesh, M)
        q_, q, u_, u, eta_, eta, b, Vq = update_SW_FE(mesh_, mesh, u_, u, eta_, eta, b)
        toc2 = clock()
        print 'Number of nodes after adaption step %d: ' % mn, len(mesh.coordinates.dat.data)
        print 'Elapsed time for adaption step %d: %1.2es' % (mn, toc2 - tic2)

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
        m_file.write(M, time = t)

    # Enter the inner timeloop:
    while cnt < rm:
        t += dt
        print 't = %1.2fs, mesh number = %d' % (t, mn)
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