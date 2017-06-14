from firedrake import *

import numpy as np
from time import clock

from utils import *

# Define initial mesh (courtesy of QMESH) and functions, with initial conditions set:
res = raw_input('Mesh type fine, medium or coarse? (f/m/c): ') or 'c'
if (res != 'f') & (res != 'm') & (res != 'c') :
    raise ValueError('Please try again, choosing f, m or c.')
mesh, Vq, q_, u_, eta_, lam_, lm_, le_, b = Tohoku_domain(res)
meshd = Meshd(mesh)
N1 = len(mesh.coordinates.dat.data)                                     # Minimum number of nodes
N2 = N1                                                                 # Maximum number of nodes
print 'Initial number of nodes : ', N1

# Choose linear or nonlinear equations:
# mode = raw_input('Linear or nonlinear equations? (l/n): ') or 'l'             # TODO: reintroduce nonlinear option
# if (mode != 'l') & (mode != 'n'):
#     raise ValueError('Please try again, choosing l or n.')

# Simulation duration:
T = float(raw_input('Simulation duration in hours (default 2)?: ') or 2.) * 3600.

# Coriolis effect test:                                                         # TODO: implement this test
cor = raw_input('Rotational or non-rotational case? (r/n)') or 'n'
if cor not in ('n', 'r') :
    raise ValueError('Please try again, choosing r or n.')

# Set up adaptivity parameters:
remesh = raw_input('Use adaptive meshing (y/n)?: ') or 'y'
if remesh == 'y' :
    hmin = float(raw_input('Minimum element size in km (default 0.5)?: ') or 0.5) * 1e3
    hmax = float(raw_input('Maximum element size in km (default 100)?: ') or 100.) * 1e3
    rm = int(raw_input('Timesteps per remesh (default 10)?: ') or 10)
    nodes = float(raw_input('Target number of nodes (default 1000)?: ') or 1000.)
    ntype = raw_input('Normalisation type? (lp/manual): ') or 'lp'
    if ntype not in ('lp', 'manual') :
        raise ValueError('Please try again, choosing lp or manual.')
    mtype = raw_input('Mesh w.r.t. speed, free surface or both? (s/f/b): ') or 'f'
    if mtype not in ('s', 'f', 'b') :
        raise ValueError('Please try again, choosing s, f or b.')
else :
    hmin = 500
    rm = int(T)
    nodes = 0
    ntype = None
    mtype = None
    if remesh != 'n':
        raise ValueError('Please try again, choosing y or n.')

# Courant number adjusted timestepping parameters:
ndump = 1
g = 9.81                                                # Gravitational acceleration (m s^{-2})
dt = 0.8 * hmin / np.sqrt(g * max(b.dat.data))          # Timestep length (s), using wavespeed sqrt(gh)
Dt = Constant(dt)
print 'Using Courant number adjusted timestep dt = %1.4f' % dt

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

        V = TensorFunctionSpace(mesh, 'CG', 1)

        if mtype != 'f' :
            # Establish velocity speed for adaption:
            spd = Function(FunctionSpace(mesh, 'CG', 1))
            spd.interpolate(sqrt(dot(u, u)))

            # Compute Hessian and metric:
            H = construct_hessian(mesh, V, spd)
            M = compute_steady_metric(mesh, V, H, spd, h_min = hmin, h_max = hmax, N = nodes, normalise = ntype)

        if mtype != 's' :

            # Compute Hessian and metric:
            H = construct_hessian(mesh, V, eta)
            M2 = compute_steady_metric(mesh, V, H, eta, h_min = hmin, h_max = hmax, N = nodes, normalise = ntype)

            if mtype == 'b' :
                M = metric_intersection(mesh, V, M, M2)

            else :
                M = M2

        # Adapt mesh and set up new function spaces:
        M.rename('Metric field')
        mesh_ = mesh
        meshd_ = Meshd(mesh_)
        tic2 = clock()
        mesh = adapt(mesh, M)
        meshd = Meshd(mesh)
        q_, q, u_, u, eta_, eta, Vq = update_SW(meshd_, meshd, u_, u, eta_, eta)
        b = update_variable(meshd_, meshd, b)
        toc2 = clock()

        # Data analysis:
        n = len(mesh.coordinates.dat.data)
        if n < N1 :
            N1 = n
        elif n > N2 :
            N2 = n

        # Print to screen:
        print ''
        print '************ Adaption step %d **************' % mn
        print 'Time = %1.2fs' % t
        print 'Number of nodes after adaption step %d: ' % mn, n
        print 'Min. nodes in mesh: %d... max. nodes in mesh: %d' % (N1, N2)
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
                m_file.write(M, time = t)
            else :
                print 't = %1.2fs' % t

# End timing and print:
toc1 = clock()
if remesh == 'y' :
    print 'Elapsed time for adaptive tank solver: %1.2fs' % (toc1 - tic1)
else :
    print 'Elapsed time for non-adaptive tank solver: %1.2fs' % (toc1 - tic1)

print 'Forward problem solved.... now for the adjoint problem.'

if remesh == 'y':

    # Reset mesh and setup:
    mesh, Vq, q_, u_, eta_, lam_, lm_, le_, b = Tohoku_domain(res)

# Set up functions of weak problem:
lam = Function(Vq)
lam.assign(lam_)
w, xi = TestFunctions(Vq)
lu, le = split(lam)
lu_, le_ = split(lam_)
luh = 0.5 * (lu + lu_)
leh = 0.5 * (le + le_)

# Set up the variational problem:
L2 = ((le - le_) * xi - Dt * g * b * inner(luh, grad(xi)) + inner(lu - lu_, w) + Dt * b * inner(grad(leh), w)) * dx
lam_prob = NonlinearVariationalProblem(L2, lam)
lam_solv = NonlinearVariationalSolver(lam_prob, solver_parameters = params)

# 'Split' functions to access their data and relabel:
lu_, le_ = lam_.split()
lu, le = lam.split()
lu.rename('Adjoint fluid velocity')
le.rename('Adjoint free surface displacement')

# Initialise counters and files:
cnt = 0
mn = 0
lam_file = File('plots/adapt_plots/tohoku_adjoint.pvd')
m_file2 = File('plots/adapt_plots/tohoku_adjoint_metric.pvd')
lam_file.write(lu, le, time = 0)