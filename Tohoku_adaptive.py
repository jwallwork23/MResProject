from firedrake import *

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from time import clock
import math
import sys

from utils import *

# Define initial mesh (courtesy of QMESH) and functions, with initial conditions set:
res = raw_input('Mesh type fine, medium or coarse? (f/m/c): ') or 'c'
if res not in ('f', 'm', 'c'):
    raise ValueError('Please try again, choosing f, m or c.')
mesh, Vq, q_, u_, eta_, lam_, lm_, le_, b = Tohoku_domain(res)
meshd = Meshd(mesh)
N1 = len(mesh.coordinates.dat.data)                                     # Minimum number of nodes
N2 = N1                                                                 # Maximum number of nodes
print 'Initial number of nodes : ', N1

# Simulation duration:
T = float(raw_input('Simulation duration in hours (default 1)?: ') or 1.) * 3600.

# Set up adaptivity parameters:
remesh = raw_input('Use adaptive meshing (y/n)?: ') or 'y'
if remesh == 'y':
    hmin = float(raw_input('Minimum element size in km (default 5)?: ') or 5.) * 1e3
    hmax = float(raw_input('Maximum element size in km (default 10)?: ') or 10.) * 1e3
    rm = int(raw_input('Timesteps per re-mesh (default 10)?: ') or 10)
    nodes = float(raw_input('Target number of nodes (default 1000)?: ') or 1000.)
    ntype = raw_input('Normalisation type? (lp/manual): ') or 'lp'
    if ntype not in ('lp', 'manual'):
        raise ValueError('Please try again, choosing lp or manual.')
    mtype = raw_input('Mesh w.r.t. speed, free surface or both? (s/f/b): ') or 'f'
    if mtype not in ('s', 'f', 'b'):
        raise ValueError('Please try again, choosing s, f or b.')
    hess_meth = raw_input('Integration by parts or double L2 projection? (parts/dL2): ') or 'dL2'
    if hess_meth not in ('parts', 'dL2'):
        raise ValueError('Please try again, choosing parts or dL2.')
else:
    hmin = 500
    rm = int(T)
    nodes = 0
    ntype = None
    mtype = None
    mat_out = 'n'
    if remesh != 'n':
        raise ValueError('Please try again, choosing y or n.')

fo = raw_input('Forward only? (y/n): ') or 'n'

# Courant number adjusted timestepping parameters:
ndump = 15
g = 9.81                                                # Gravitational acceleration (m s^{-2})
dt = 0.8 * hmin / np.sqrt(g * max(b.dat.data))          # Timestep length (s), using wavespeed sqrt(gh)
Dt = Constant(dt)
print 'Using Courant number adjusted timestep dt = %1.4f' % dt

# Gauge locations:
gloc = {'P02': lonlat2tangent_pair(142.5, 38.5, 143, 37),
        'P06': lonlat2tangent_pair(142.6, 38.7, 143, 37),
        '801': lonlat2tangent_pair(141.7, 38.2, 143, 37),
        '802': lonlat2tangent_pair(142.1, 39.3, 143, 37),
        '803': lonlat2tangent_pair(141.8, 38.9, 143, 37),
        '804': lonlat2tangent_pair(142.2, 39.7, 143, 37),
        '806': lonlat2tangent_pair(141.2, 37.0, 143, 37)}

# Set gauge arrays:
gtype = raw_input('Pressure or tide gauge? (p/t): ') or 'p'
if gtype == 'p':
    gauge = raw_input('Gauge P02 or P06?: ') or 'P02'
    gcoord = gloc[gauge]
elif gtype == 't':
    gauge = raw_input('Gauge 801, 802, 803, 804 or 806?: ') or '801'
    gcoord = gloc[gauge]

# Set up functions of the weak problem:
q = Function(Vq)
q.assign(q_)
u, eta = q.split()

# Specify solver parameters:
params = {'mat_type': 'matfree',
          'snes_type': 'ksponly',
          'pc_type': 'python',
          'pc_python_type': 'firedrake.AssembledPC',
          'assembled_pc_type': 'lu',
          'snes_lag_preconditioner': -1,
          'snes_lag_preconditioner_persists': True}

# Initialise counters, files and arrays:
t = 0.
cnt = 0
dumpn = 0
mn = 0
u.rename('Fluid velocity')
eta.rename('Free surface displacement')
q_file = File('plots/adapt_plots/tohoku_adapt.pvd')
q_file.write(u, eta, time=t)
gauge_dat = [eta.at(gcoord)]
maxi = max(eta.at(gloc['801']), eta.at(gloc['802']), eta.at(gloc['803']), eta.at(gloc['804']), eta.at(gloc['806']), 0.5)
damage_measure = [math.log(maxi)]
tic1 = clock()

while t < T - 0.5 * dt:

    # Increment counters:
    cnt += 1
    t += dt
    dumpn += 1

    if (remesh == 'y') & (cnt % rm == 0):

        mn += 1

        # Compute Hessian and metric:
        tic2 = clock()
        V = TensorFunctionSpace(mesh, 'CG', 1)
        if mtype != 'f':
            spd = Function(FunctionSpace(mesh, 'CG', 1))        # Fluid speed
            spd.interpolate(sqrt(dot(u, u)))
            H = construct_hessian(mesh, V, spd, method=hess_meth)
            M = compute_steady_metric(mesh, V, H, spd, h_min=hmin, h_max=hmax, N=nodes, normalise=ntype)
        if mtype != 's':
            H = construct_hessian(mesh, V, eta, method=hess_meth)
            M2 = compute_steady_metric(mesh, V, H, eta, h_min=hmin, h_max=hmax, N=nodes, normalise=ntype)
            if mtype == 'b':
                M = metric_intersection(mesh, V, M, M2)
            else:
                M = M2
        adaptor = AnisotropicAdaptation(mesh, M)
        mesh = adaptor.adapted_mesh

        # Interpolate functions onto new mesh:
        u, u_, eta, eta_, q, q_, b, W = interp_Taylor_Hood(adaptor, u, u_, eta, eta_, b)

        toc2 = clock()

        # Data analysis:
        n = len(mesh.coordinates.dat.data)
        if n < N1:
            N1 = n
        elif n > N2:
            N2 = n

        # Print to screen:
        print ''
        print '************ Adaption step %d **************' % mn
        print 'Time = %1.2fs' % t
        print 'Number of nodes after adaption step %d: ' % mn, n
        print 'Min. nodes in mesh: %d... max. nodes in mesh: %d' % (N1, N2)
        print 'Elapsed time for this step: %1.2fs' % (toc2 - tic2)
        print ''

    # Set up functions of weak problem:
    v, ze = TestFunctions(Vq)
    u, eta = split(q)
    u_, eta_ = split(q_)
    uh = 0.5 * (u + u_)
    etah = 0.5 * (eta + eta_)

    # Set up the variational problem
    L = (ze * (eta - eta_) - Dt * inner(b * uh, grad(ze)) + inner(u - u_, v) + Dt * g * (inner(grad(etah), v))) * dx
    q_prob = NonlinearVariationalProblem(L, q)
    q_solv = NonlinearVariationalSolver(q_prob, solver_parameters=params)

    # Split to access data and relabel functions:
    u_, eta_ = q_.split()
    u, eta = q.split()
    u.rename('Fluid velocity')
    eta.rename('Free surface displacement')

    # Solve the problem and update:
    q_solv.solve()
    q_.assign(q)

    if t < T:
        gauge_dat.append(eta.at(gcoord))
        maxi = max(eta.at(gloc['801']), eta.at(gloc['802']), eta.at(gloc['803']), eta.at(gloc['804']),
                   eta.at(gloc['806']), 0.5)
        damage_measure.append(math.log(maxi))

    if dumpn == ndump:
        dumpn -= ndump
        q_file.write(u, eta, time=t)
        if remesh == 'n':
            print 't = %1.2fs' % t

# End timing and print:
toc1 = clock()
if remesh == 'y':
    print 'Elapsed time for adaptive forward solver: %1.2fs' % (toc1 - tic1)
else:
    print 'Elapsed time for non-adaptive forward solver: %1.2fs' % (toc1 - tic1)

# Plot gauge time series:
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(np.linspace(0, 60, len(gauge_dat)), gauge_dat)
plt.gcf().subplots_adjust(bottom=0.15)
plt.ylim([-5, 5])
# plt.legend()
plt.xlabel(r'Time elapsed (mins)')
plt.ylabel(r'Free surface (m)')
plt.savefig('plots/tsunami_outputs/screenshots/adaptive_gauge_timeseries_{y}.png'.format(y=gauge))

gauge_timeseries(gauge, gauge_dat)

# Plot damage measures time series:
plt.clf()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(np.linspace(0, 60, len(damage_measure)), damage_measure)
plt.gcf().subplots_adjust(bottom=0.15)
plt.axis([0, 60, -1.5, 3.5])
plt.axhline(-1, linestyle='--', color='blue')
plt.axhline(0, linestyle='--', color='green')
plt.axhline(1, linestyle='--', color='yellow')
plt.axhline(2, linestyle='--', color='orange')
plt.axhline(3, linestyle='--', color='red')
plt.xlabel(r'Time elapsed (mins)')
plt.ylabel(r'Maximal log free surface')
plt.savefig('plots/tsunami_outputs/screenshots/damage_measure_timeseries.png')

if fo == 'y':
    sys.exit(1)

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

f = Function(Ve, name='Forcing term')
f.interpolate(Expression('(x[0] >= 1e4) & (x[0] <= 2.5e4) & (x[1] >= 1.8e5) & (x[1] <= 2.2e5) ? 1. : 0.'))

# Set up the variational problem:                                   TODO: Check this forcing term below works
L2 = ((le - le_) * xi - Dt * g * b * inner(luh, grad(xi)) + f * xi
      + inner(lu - lu_, w) + Dt * b * inner(grad(leh), w)) * dx
lam_prob = NonlinearVariationalProblem(L2, lam)
lam_solv = NonlinearVariationalSolver(lam_prob, solver_parameters=params)

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
lam_file.write(lu, le, time=0)
tic3 = clock()

while t > 0.5 * dt:

    mn += 1
    cnt = 0

    if remesh == 'y':

        # Compute Hessian and metric:
        V = TensorFunctionSpace(mesh, 'CG', 1)
        if mtype != 'f':
            lspd = Function(FunctionSpace(mesh, 'CG', 1))
            lspd.interpolate(sqrt(dot(lu, lu)))
            H = construct_hessian(mesh, V, lspd)
            M = compute_steady_metric(mesh, V, H, lspd, h_min=hmin, h_max=hmax, N=nodes, normalise=ntype)
        if mtype != 's':
            H = construct_hessian(mesh, V, le)
            M2 = compute_steady_metric(mesh, V, H, le, h_min=hmin, h_max=hmax, N=nodes, normalise=ntype)
            if mtype == 'b':
                M = metric_intersection(mesh, V, M, M2)
            else:
                M = M2

        # Adapt mesh and set up new function spaces:
        M.rename('Adjoint metric field')
        mesh_ = mesh
        meshd_ = Meshd(mesh_)
        tic4 = clock()
        mesh = adapt(mesh, M)
        meshd = Meshd(mesh)
        lam_, lam, lu_, lu, le_, le, Vq = update_SW(meshd_, meshd, lu_, lu, le_, le)
        b = update_variable(meshd_, meshd, b)
        f = update_variable(meshd_, meshd, f)
        toc4 = clock()

        # Data analysis:
        n = len(mesh.coordinates.dat.data)
        if n < N1:
            N1 = n
        elif n > N2:
            N2 = n

        # Print to screen:
        print ''
        print '************ Adaption step %d **************' % mn
        print 'Time = %1.2fs' % t
        print 'Number of nodes after adaption step %d: ' % mn, n
        print 'Min. nodes in mesh: %d... max. nodes in mesh: %d' % (N1, N2)
        print 'Elapsed time for adaption step %d: %1.2es' % (mn, toc4 - tic4)
        print ''

    # Set up functions of weak problem:
    w, xi = TestFunctions(Vq)
    lu, le = split(lam)
    lu_, le_ = split(lam_)
    luh = 0.5 * (lu + lu_)
    leh = 0.5 * (le + le_)

    # Set up the variational problem
    L2 = ((le - le_) * xi - Dt * g * b * inner(luh, grad(xi)) + inner(lu - lu_, w) + Dt * b * inner(grad(leh), w)) * dx
    lam_prob = NonlinearVariationalProblem(L2, lam)
    lam_solv = NonlinearVariationalSolver(lam_prob, solver_parameters=params)

    # 'Split' functions to access their data and relabel:
    lu_, le_ = lam_.split()
    lu, le = lam.split()
    lu.rename('Adjoint fluid velocity')
    le.rename('Adjoint free surface displacement')

    # Enter the inner timeloop:
    if remesh == 'y':
        while cnt < rm:
            t -= dt
            cnt += 1
            lam_solv.solve()
            lam_.assign(lam)
            dumpn -= 1

            if dumpn == 0:
                dumpn += ndump
                lam_file.write(lu, le, time=T-t)
    else:
        while t > 0:
            t -= dt
            print 't = %1.2fs' % t
            lam_solv.solve()
            lam_.assign(lam)
            dumpn -= 1

            if dumpn == 0:
                dumpn += ndump
                lam_file.write(lu, le, time=T-t)

# End timing and print:
toc3 = clock()
if remesh == 'y':
    print 'Elapsed time for adaptive adjoint solver: %1.2fs' % (toc3 - tic3)
else:
    print 'Elapsed time for non-adaptive adjoint solver: %1.2fs' % (toc3 - tic3)
