from firedrake import *

import numpy as np
from time import clock

from utils import adapt, construct_hessian, compute_steady_metric, interp, Meshd, Tohoku_domain, update_SW_FE

# Define initial mesh (courtesy of QMESH) and functions, with initial conditions set:
res = raw_input('Mesh type fine, medium or coarse? (f/m/c): ') or 'c'
if (res != 'f') & (res != 'm') & (res != 'c') :
    raise ValueError('Please try again, choosing f, m or c.')
mesh, Vq, q_, u_, eta_, lam_, lm_, le_, b = Tohoku_domain(res)
meshd = Meshd(mesh)
print 'Initial number of nodes : ', len(mesh.coordinates.dat.data)

# Choose linear or nonlinear equations:
# mode = raw_input('Linear or nonlinear equations? (l/n): ') or 'l'             # TODO: reintroduce nonlinear option
# if (mode != 'l') & (mode != 'n'):
#     raise ValueError('Please try again, choosing l or n.')

# Specify timestepping parameters:
dt = float(raw_input('Timestep in seconds (default 15)?: ') or 15)              # TODO: consider adaptive timestepping?
Dt = Constant(dt)
T = float(raw_input('Simulation duration in hours (default 2)?: ') or 2.) * 3600.
ndump = 1

# Set up adaptivity parameters:
remesh = raw_input('Use adaptive meshing (y/n)?: ') or 'y'
if remesh == 'y' :
    hmin = float(raw_input('Minimum element size in km (default 0.5)?: ') or 0.5) * 1e3
    rm = int(raw_input('Timesteps per remesh (default 4)?: ') or 4)
    nodes = float(raw_input('Target number of nodes (default 1000)?: ') or 1000.)
    ntype = raw_input('Normalisation type? (lp/manual): ') or 'lp'
else :
    hmin = 0
    rm = int(T / dt)
    nodes = 0
    ntype = None
    if remesh != 'n':
        raise ValueError('Please try again, choosing y or n.')

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
g = 9.81            # Gravitational acceleration (m s^{-2})
L = (ze * (eta - eta_) - Dt * inner(b * uh, grad(ze)) +
         inner(u - u_, v) + Dt * g * (inner(grad(etah), v))) * dx
q_prob = NonlinearVariationalProblem(L, q)
q_solv = NonlinearVariationalSolver(q_prob, solver_parameters = params)

# 'Split' functions in order to access their data and then relabel:
u_, eta_ = q_.split()
u, eta = q.split()
u.rename('Fluid velocity')
eta.rename('Free surface displacement')

# Initialise counters and files:
t = 0.
mn = 0
dumpn = 0
q_file = File('plots/adapt_plots/tohoku_adapt.pvd')
m_file = File('plots/adapt_plots/tohoku_adapt_metric.pvd')
q_file.write(u, eta, time = t)
tic1 = clock()

while t < T - 0.5 * dt:

    mn += 1
    cnt = 0

    if remesh == 'y' :

        # Compute Hessian and metric:
        V = TensorFunctionSpace(mesh, 'CG', 1)
        H = construct_hessian(mesh, V, eta)
        M = compute_steady_metric(mesh, V, H, eta, h_min = hmin, N = nodes)
        M.rename('Metric field')

        # Adapt mesh and update FE setup:
        mesh_ = mesh
        meshd_ = Meshd(mesh_)
        tic2 = clock()
        mesh = adapt(mesh, M)
        meshd = Meshd(mesh)
        q_, q, u_, u, eta_, eta, b, Vq = update_SW_FE(meshd_, meshd, u_, u, eta_, eta, b)
        toc2 = clock()

        # Print to screen:
        print ''
        print '************ Adaption step %d **************' % mn
        print 'Time = %1.2fs' % t
        print 'Number of nodes after adaption step %d: ' % mn, len(mesh.coordinates.dat.data)
        print 'Elapsed time for adaption step %d: %1.2es' % (mn, toc2 - tic2)
        print ''

    # Set up functions of weak problem:
    v, ze = TestFunctions(Vq)
    u, eta = split(q)
    u_, eta_ = split(q_)
    uh = 0.5 * (u + u_)
    etah = 0.5 * (eta + eta_)

    # Set up the variational problem
    L = (ze * (eta - eta_) - Dt * inner(b * uh, grad(ze)) +
         inner(u - u_, v) + Dt * g * (inner(grad(etah), v))) * dx
    q_prob = NonlinearVariationalProblem(L, q)
    q_solv = NonlinearVariationalSolver(q_prob, solver_parameters = params)

    # 'Split' functions to access their data and relabel:
    u_, eta_ = q_.split()
    u, eta = q.split()
    u.rename('Fluid velocity')
    eta.rename('Free surface displacement')

    # Enter the inner timeloop:
    while cnt < rm :
        t += dt
        cnt += 1
        q_solv.solve()
        q_.assign(q)
        dumpn += 1
        if dumpn == ndump :

            dumpn -= ndump
            q_file.write(u, eta, time = t)

            if remesh == 'y' :
                m_file.write(M, time=t)
            else :
                print 't = %1.2fs, mesh number =' % t

# End timing and print:
toc1 = clock()
if remesh == 'y' :
    print 'Elapsed time for adaptive tank solver: %1.2es' % (toc1 - tic1)
else :
    print 'Elapsed time for non-adaptive tank solver: %1.2es' % (toc1 - tic1)