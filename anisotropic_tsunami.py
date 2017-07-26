from firedrake import *
import numpy as np
from time import clock
import math
import sys

from utils.adaptivity import compute_steady_metric, construct_hessian
from utils.conversion import from_latlon
from utils.domain import Tohoku_domain
from utils.interp import interp_Taylor_Hood
from utils.storage import gauge_timeseries

# Change backend to resolve framework problems:
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

print ''
print '******************************** ANISOTROPIC ADAPTIVE TSUNAMI SIMULATION ********************************'
print ''
print 'Options...'

# Define initial mesh (courtesy of QMESH) and functions, with initial conditions set:
coarseness = int(raw_input('Mesh coarseness? (Integer in range 1-5, default 4): ') or 4)
mesh, W, q_, u_, eta_, lam_, lm_, le_, b = Tohoku_domain(coarseness)
N1 = len(mesh.coordinates.dat.data)                                     # Minimum number of nodes
N2 = N1                                                                 # Maximum number of nodes
print '...... mesh loaded. Initial number of vertices : ', N1

# Set target number of vertices:
nodes = 0.1 * N1

# Set physical parameters:
g = 9.81                        # Gravitational acceleration (m s^{-2})

# Simulation duration:
T = float(raw_input('Simulation duration in hours (default 1)?: ') or 1.) * 3600.

# Set up adaptivity parameters:
hmin = float(raw_input('Minimum element size in km (default 0.5)?: ') or 0.5) * 1e3
hmax = float(raw_input('Maximum element size in km (default 10000)?: ') or 10000.) * 1e3
rm = int(raw_input('Timesteps per re-mesh (default 10)?: ') or 10)
ntype = raw_input('Normalisation type? (lp/manual): ') or 'lp'
if ntype not in ('lp', 'manual'):
    raise ValueError('Please try again, choosing lp or manual.')
mtype = raw_input('Mesh w.r.t. speed, free surface or both? (s/f/b): ') or 'f'
if mtype not in ('s', 'f', 'b'):
    raise ValueError('Please try again, choosing s, f or b.')
hess_meth = raw_input('Integration by parts or double L2 projection? (parts/dL2): ') or 'dL2'
if hess_meth not in ('parts', 'dL2'):
    raise ValueError('Please try again, choosing parts or dL2.')

# Courant number adjusted timestepping parameters:
dt = float(raw_input('Specify timestep in seconds (default 1): ') or 1.)
Dt = Constant(dt)
cdt = hmin / np.sqrt(g * max(b.dat.data))
if dt > cdt:
    print 'WARNING: chosen timestep dt =', dt, 'exceeds recommended value of', cdt
    if raw_input('Are you happy to proceed? (y/n)') == 'n':
        exit(23)
ndump = int(60. / dt)

# Convert gauge locations:
glatlon = {'P02': (38.5, 142.5), 'P06': (38.7, 142.6),
           '801': (38.2, 141.7), '802': (39.3, 142.1), '803': (38.9, 141.8), '804': (39.7, 142.2), '806': (37.0, 141.2)}
gloc = {}
for key in glatlon:
    east, north, zn, zl = from_latlon(glatlon[key][0], glatlon[key][1], force_zone_number=54)
    gloc[key] = (east, north)

# Set gauge arrays:
gtype = raw_input('Pressure or tide gauge? (p/t): ') or 'p'
if gtype == 'p':
    gauge = raw_input('Gauge P02 or P06?: ') or 'P02'
    gcoord = gloc[gauge]
elif gtype == 't':
    gauge = raw_input('Gauge 801, 802, 803, 804 or 806?: ') or '801'
    gcoord = gloc[gauge]
else:
    ValueError('Gauge type not recognised. Please choose p or t.')
dm = raw_input('Evaluate damage measures? (y/n, default n): ') or 'n'

# Set up functions of the weak problem:
q = Function(W)
q.assign(q_)
u, eta = q.split()

# Initialise counters, files and gauge data measurements:
t = 0.
dumpn = 0
mn = 0
u.rename('Fluid velocity')
eta.rename('Free surface displacement')
q_file = File('plots/adapt_plots/tohoku_adapt.pvd')
q_file.write(u, eta, time=t)
gauge_dat = [eta.at(gcoord)]
if dm == 'y':
    damage_measure = [math.log(max(eta.at(gloc['801']), eta.at(gloc['802']), eta.at(gloc['803']),
                                   eta.at(gloc['804']), eta.at(gloc['806']), 0.5))]
print ''
print 'Entering outer timeloop!'
tic1 = clock()
while t < T - 0.5 * dt:
    mn += 1

    # Compute Hessian and metric:
    tic2 = clock()
    V = TensorFunctionSpace(mesh, 'CG', 1)
    if mtype != 'f':
        spd = Function(FunctionSpace(mesh, 'CG', 1))        # Fluid speed
        spd.interpolate(sqrt(dot(u, u)))
        H = construct_hessian(mesh, V, spd, method=hess_meth)
        M = compute_steady_metric(mesh, V, H, spd, h_min=hmin, h_max=hmax, num=nodes, normalise=ntype)
    if mtype != 's':
        H = construct_hessian(mesh, V, eta, method=hess_meth)
        M2 = compute_steady_metric(mesh, V, H, eta, h_min=hmin, h_max=hmax, num=nodes, normalise=ntype)
        if mtype == 'b':
            M = metric_intersection(mesh, V, M, M2)
        else:
            M = M2
    adaptor = AnisotropicAdaptation(mesh, M)
    mesh = adaptor.adapted_mesh

    # Interpolate functions onto new mesh:
    u, u_, eta, eta_, q, q_, b, W = interp_Taylor_Hood(mesh, u, u_, eta, eta_, b)
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
    v, ze = TestFunctions(W)
    u, eta = split(q)
    u_, eta_ = split(q_)
    uh = 0.5 * (u + u_)
    etah = 0.5 * (eta + eta_)

    # Set up the variational problem
    L = (ze * (eta - eta_) - Dt * inner(b * uh, grad(ze)) + inner(u - u_, v) + Dt * g * (inner(grad(etah), v))) * dx
    q_prob = NonlinearVariationalProblem(L, q)
    q_solv = NonlinearVariationalSolver(q_prob, solver_parameters={'mat_type': 'matfree',
                                                                   'snes_type': 'ksponly',
                                                                   'pc_type': 'python',
                                                                   'pc_python_type': 'firedrake.AssembledPC',
                                                                   'assembled_pc_type': 'lu',
                                                                   'snes_lag_preconditioner': -1,
                                                                   'snes_lag_preconditioner_persists': True})
    # Split to access data and relabel functions:
    u_, eta_ = q_.split()
    u, eta = q.split()
    u.rename('Fluid velocity')
    eta.rename('Free surface displacement')

    for j in range(rm):
        t += dt
        dumpn += 1

        # Solve the problem and update:
        q_solv.solve()
        q_.assign(q)

        # Store data:
        gauge_dat.append(eta.at(gcoord))
        if dm == 'y':
            damage_measure.append(math.log(max(eta.at(gloc['801']), eta.at(gloc['802']), eta.at(gloc['803']),
                                               eta.at(gloc['804']), eta.at(gloc['806']), 0.5)))
        if dumpn == ndump:
            dumpn -= ndump
            q_file.write(u, eta, time=t)
print '\a'
# End timing and print:
toc1 = clock()
print 'Elapsed time for adaptive forward solver: %1.2f mins' % ((toc1 - tic1) / 60.)

# Plot gauge time series:
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(np.linspace(0, 60, len(gauge_dat)), gauge_dat)
plt.gcf().subplots_adjust(bottom=0.15)
plt.ylim([-5, 5])
plt.xlabel(r'Time elapsed (mins)')
plt.ylabel(r'Free surface (m)')
plt.savefig('plots/tsunami_outputs/screenshots/anisotropic_gauge_timeseries_{y}.png'.format(y=gauge))

# Store gauge timeseries data to file:
gauge_timeseries(gauge, gauge_dat)

# Plot damage measures time series:
if dm == 'y':
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
    plt.savefig('plots/tsunami_outputs/screenshots/anisotropic_damage_measure_timeseries.png')
