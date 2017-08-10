from firedrake import *
import numpy as np
from time import clock
import math
import sys

from utils.adaptivity import compute_steady_metric, construct_hessian, metric_intersection
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
print ''
print 'Mesh adaptive solver initially defined on a mesh of',

# Define initial mesh (courtesy of QMESH) and functions, with initial conditions set:
coarseness = int(raw_input('coarseness (Integer in range 1-5, default 5): ') or 5)
mesh, W, q_, u_, eta_, lam_, lu_, le_, b = Tohoku_domain(coarseness)
N1 = len(mesh.coordinates.dat.data)                                     # Minimum number of vertices
N2 = N1                                                                 # Maximum number of vertices
SumN = N1                                                               # Sum over vertex counts
print '...... mesh loaded. Initial number of vertices : ', N1

# Set physical parameters:
g = 9.81                        # Gravitational acceleration (m s^{-2})

print 'More options...'
numVer = float(raw_input('Target vertex count as a proportion of the initial number? (default 0.85): ') or 0.85) * N1
hmin = float(raw_input('Minimum element size in km (default 0.5)?: ') or 0.5) * 1e3
hmax = float(raw_input('Maximum element size in km (default 10000)?: ') or 10000.) * 1e3
ntype = raw_input('Normalisation type? (lp/manual, default lp): ') or 'lp'
if ntype not in ('lp', 'manual'):
    raise ValueError('Please try again, choosing lp or manual.')
mtype = raw_input('Mesh w.r.t. speed, free surface or both? (s/f/b, default b): ') or 'b'
if mtype not in ('s', 'f', 'b'):
    raise ValueError('Please try again, choosing s, f or b.')
mat_out = bool(raw_input('Hit any key to output Hessian and metric: ')) or False
iso = bool(raw_input('Hit anything but enter to use isotropic, rather than anisotropic: ')) or False
if not iso:
    hess_meth = raw_input('Integration by parts or double L2 projection? (parts/dL2, default dL2): ') or 'dL2'
    if hess_meth not in ('parts', 'dL2'):
        raise ValueError('Please try again, choosing parts or dL2.')

# Courant number adjusted timestepping parameters:
T = float(raw_input('Simulation duration in minutes (default 25)?: ') or 25.) * 60.
dt = float(raw_input('Specify timestep in seconds (default 1): ') or 1.)
Dt = Constant(dt)
cdt = hmin / np.sqrt(g * max(b.dat.data))
if dt > cdt:
    print 'WARNING: chosen timestep dt =', dt, 'exceeds recommended value of', cdt
    if raw_input('Are you happy to proceed? (y/n)') == 'n':
        exit(23)
ndump = int(60. / dt)           # Timesteps per data dump
rm = int(raw_input('Timesteps per re-mesh (default 30)?: ') or 30)

# Convert gauge locations to UTM coordinates:
glatlon = {'P02': (38.5002, 142.5016), 'P06': (38.6340, 142.5838),
           '801': (38.2, 141.7), '802': (39.3, 142.1), '803': (38.9, 141.8), '804': (39.7, 142.2), '806': (37.0, 141.2)}
gloc = {}
for key in glatlon:
    east, north, zn, zl = from_latlon(glatlon[key][0], glatlon[key][1], force_zone_number=54)
    gloc[key] = (east, north)

# Set gauge arrays:
gtype = raw_input('Pressure or tide gauge? (p/t, default p): ') or 'p'
if gtype == 'p':
    gauge = raw_input('Gauge P02 or P06? (default P02): ') or 'P02'
    gcoord = gloc[gauge]
elif gtype == 't':
    gauge = raw_input('Gauge 801, 802, 803, 804 or 806? (default 801): ') or '801'
    gcoord = gloc[gauge]
else:
    ValueError('Gauge type not recognised. Please choose p or t.')

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
if iso:
    q_file = File('plots/isotropic_outputs/tsunami.pvd')
    if mat_out:
        m_file = File('plots/isotropic_outputs/tsunami_metric.pvd')
        h_file = File('plots/isotropic_outputs/tsunami_hessian.pvd')
else:
    q_file = File('plots/anisotropic_outputs/tsunami.pvd')
    if mat_out:
        m_file = File('plots/anisotropic_outputs/tsunami_metric.pvd')
        h_file = File('plots/anisotropic_outputs/tsunami_hessian.pvd')
q_file.write(u, eta, time=0)
gauge_dat = [eta.at(gcoord)]
print ''
print 'Entering outer timeloop!'
tic1 = clock()
while t < T - 0.5 * dt:
    mn += 1

    # Compute Hessian and metric:
    tic2 = clock()
    V = TensorFunctionSpace(mesh, 'CG', 1)
    H = Function(V)
    if mtype != 'f':
        spd = Function(FunctionSpace(mesh, 'CG', 1))        # Fluid speed
        spd.interpolate(sqrt(dot(u, u)))
        if iso:
            for i in range(len(H.dat.data)):
                H.dat.data[i][0, 0] = spd.dat.data[i]
                H.dat.data[i][1, 1] = spd.dat.data[i]
        else:
            H = construct_hessian(mesh, V, spd, method=hess_meth)
        M = compute_steady_metric(mesh, V, H, spd, h_min=hmin, h_max=hmax, num=numVer, normalise=ntype)
    if mtype != 's':
        if iso:
            for i in range(len(H.dat.data)):
                H.dat.data[i][0, 0] = np.abs(eta.dat.data[i])
                H.dat.data[i][1, 1] = np.abs(eta.dat.data[i])
        else:
            H = construct_hessian(mesh, V, eta, method=hess_meth)
        M2 = compute_steady_metric(mesh, V, H, eta, h_min=hmin, h_max=hmax, num=numVer, normalise=ntype)
        if mtype == 'b':
            M = metric_intersection(mesh, V, M, M2)
        else:
            M = Function(V)
            M.assign(M2)
    adaptor = AnisotropicAdaptation(mesh, M)
    mesh = adaptor.adapted_mesh

    # Interpolate functions onto new mesh:
    u, u_, eta, eta_, q, q_, b, W = interp_Taylor_Hood(mesh, u, u_, eta, eta_, b)

    # Mesh resolution analysis:
    n = len(mesh.coordinates.dat.data)
    SumN += n
    if n < N1:
        N1 = n
    elif n > N2:
        N2 = n

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

    # Inner timeloop:
    for j in range(rm):
        t += dt
        dumpn += 1

        # Solve the problem and update:
        q_solv.solve()
        q_.assign(q)

        # Store data:
        gauge_dat.append(eta.at(gcoord))
        if dumpn == ndump:
            dumpn -= ndump
            q_file.write(u, eta, time=t)
            if mat_out:
                H.rename('Hessian')
                M.rename('Metric')
                h_file.write(H, time=t)
                m_file.write(M, time=t)
    toc2 = clock()

    # Print to screen:
    print ''
    print '************ Adaption step %d **************' % mn
    print 'Time = %1.2f mins / %1.1f mins' % (t / 60., T / 60.)
    print 'Number of vertices after adaption step %d: ' % mn, n
    print 'Min/max vertex counts: %d, %d' % (N1, N2)
    print 'Mean vertex count: %d' % (float(SumN) / mn)
    print 'Elapsed time for this step: %1.2fs' % (toc2 - tic2)
    print ''
print '\a'
toc1 = clock()
print 'Elapsed time for adaptive solver: %1.1fs (%1.2f mins)' % (toc1 - tic1, (toc1 - tic1) / 60)

# Store gauge timeseries data to file:
gauge_timeseries(gauge, gauge_dat)