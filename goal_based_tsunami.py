from firedrake import *
import numpy as np
from time import clock
import math
import sys

from utils.adaptivity import compute_steady_metric, construct_hessian
from utils.conversion import from_latlon
from utils.domain import Tohoku_domain
from utils.interp import interp, interp_Taylor_Hood
from utils.storage import gauge_timeseries

# Change backend to resolve framework problems:
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

print ''
print '******************************** GOAL-BASED ADAPTIVE TSUNAMI SIMULATION ********************************'
print ''
print 'GOAL-BASED, mesh adaptive solver initially defined on a mesh of',
tic1 = clock()

# Define initial mesh (courtesy of QMESH) and functions, with initial conditions set:
coarseness = int(raw_input('coarseness (Integer in range 1-5, default 5): ') or 5)
mesh, W, q_, u_, eta_, lam_, lu_, le_, b = Tohoku_domain(coarseness)
mesh_ = mesh
N1 = len(mesh.coordinates.dat.data)                                     # Minimum number of vertices
N2 = N1                                                                 # Maximum number of vertices
print '...... mesh loaded. Initial number of vertices : ', N1

# Set up adaptivity parameters:
print 'More options...'
hmin = float(raw_input('Minimum element size in km (default 0.5)?: ') or 0.5) * 1e3
hmax = float(raw_input('Maximum element size in km (default 10000)?: ') or 10000.) * 1e3
ntype = raw_input('Normalisation type? (lp/manual): ') or 'lp'
if ntype not in ('lp', 'manual'):
    raise ValueError('Please try again, choosing lp or manual.')
hess_meth = raw_input('Integration by parts or double L2 projection? (parts/dL2, default dL2): ') or 'dL2'
if hess_meth not in ('parts', 'dL2'):
    raise ValueError('Please try again, choosing parts or dL2.')
nodes = 0.5 * N1                # Target number of vertices

# Specify parameters:
T = float(raw_input('Simulation duration in minutes (default 25)?: ') or 25.) * 60.
Ts = 5. * 60.                   # Time range lower limit (s), during which we can assume the wave won't reach the shore
g = 9.81                        # Gravitational acceleration (m s^{-2})
dt = float(raw_input('Specify timestep in seconds (default 1): ') or 1.)
Dt = Constant(dt)
ndump = int(60. / dt)           # Timesteps per data dump
rm = int(raw_input('Timesteps per re-mesh (default 60)?: ') or 60)
stored = raw_input('Adjoint already computed? (y/n, default n): ') or 'n'

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

# Specify solver parameters:
params = {'mat_type': 'matfree',
          'snes_type': 'ksponly',
          'pc_type': 'python',
          'pc_python_type': 'firedrake.AssembledPC',
          'assembled_pc_type': 'lu',
          'snes_lag_preconditioner': -1,
          'snes_lag_preconditioner_persists': True}

# Initalise counters:
t = T
i = -1
dumpn = ndump
meshn = rm

# Forcing switch:
switch = Constant(1.)
switched = 'on'

if stored == 'n':
    # Establish indicator function for adjoint equations:
    f = Function(W.sub(1), name='Forcing term')
    f.interpolate(Expression('(x[0] > 490e3) & (x[0] < 580e3) & (x[1] > 4130e3) & (x[1] < 4260e3) ? 1. : 0.'))

    # Set up dependent variables of the adjoint problem:
    lam = Function(W)
    lam.assign(lam_)
    lu, le = lam.split()
    lu.rename('Adjoint velocity')
    le.rename('Adjoint free surface')

    # Interpolate velocity onto P1 space and store final time data to HDF5 and PVD:
    lu_P1 = Function(VectorFunctionSpace(mesh, 'CG', 1), name='P1 adjoint velocity')
    lu_P1.interpolate(lu)
    with DumbCheckpoint('data_dumps/tsunami/adjoint_soln_{y}'.format(y=i), mode=FILE_CREATE) as chk:
        chk.store(lu_P1)
        chk.store(le)
    lam_file = File('plots/goal-based_outputs/tsunami_adjoint.pvd')
    lam_file.write(lu, le, time=T)

    # Establish test functions and midpoint averages:
    w, xi = TestFunctions(W)
    lu, le = split(lam)
    lu_, le_ = split(lam_)
    luh = 0.5 * (lu + lu_)
    leh = 0.5 * (le + le_)

    # Set up the variational problem:
    La = ((le - le_) * xi - Dt * g * inner(luh, grad(xi)) - switch * f * xi
          + inner(lu - lu_, w) + Dt * (b * inner(grad(leh), w) + leh * inner(grad(b), w))) * dx
    lam_prob = NonlinearVariationalProblem(La, lam)
    lam_solv = NonlinearVariationalSolver(lam_prob, solver_parameters=params)

    # Split to access data:
    lu, le = lam.split()
    lu_, le_ = lam_.split()

    print ''
    print 'Starting fixed resolution adjoint run...'
    tic2 = clock()
while t > Ts + 0.5 * dt:

    # Increment counters:
    t -= dt
    dumpn -= 1
    meshn -= 1

    # Remove forcing term:                          TODO: smoothen f in space and in time
    if (t < Ts + 0.5 * dt) & (switched == 'on'):
        switched = 'off'
        switch.assign(0)

    # Solve the problem and update:
    if stored == 'n':
        lam_solv.solve()
        lam_.assign(lam)

        # Dump to vtu:
        if dumpn == 0:
            dumpn += ndump
            lam_file.write(lu, le, time=t)

        # Dump to HDF5:
        if meshn == 0:
            meshn += rm
            i -= 1
            # Interpolate velocity onto P1 space and store final time data to HDF5 and PVD:
            if stored == 'n':
                print 't = %1.1fs' % t
                lu_P1.interpolate(lu)
                with DumbCheckpoint('data_dumps/tsunami/adjoint_soln_{y}'.format(y=i), mode=FILE_CREATE) as chk:
                    chk.store(lu_P1)
                    chk.store(le)
if stored == 'n':
    print '... done!',
    toc2 = clock()
    print 'Elapsed time for adjoint solver: %1.2fs' % (toc2 - tic2)

# Set up dependent variables of the forward problem:
q = Function(W)
q.assign(q_)
u, eta = q.split()

# Label variables:
u.rename('Fluid velocity')
eta.rename('Free surface displacement')

# Intialise files:
q_file = File('plots/goal-based_outputs/tsunami_forward.pvd')
q_file.write(u, eta, time=0)
sig_file = File('plots/goal-based_outputs/tsunami_significance.pvd')
gauge_dat = [eta.at(gcoord)]

# Initialise counters:
t = 0.
dumpn = 0
i0 = i

print ''
print 'Starting mesh adaptive forward run...'
while t < T - 0.5 * dt:
    tic2 = clock()

    # Interpolate velocity in a P1 space:
    vel = Function(VectorFunctionSpace(mesh, 'CG', 1))
    vel.interpolate(u)

    # Create functions to hold inner product and significance data:
    ip = Function(W.sub(1), name='Inner product')
    significance = Function(W.sub(1), name='Significant regions')

    # Take maximal L2 inner product as most significant:
    for j in range(max(i, int((Ts - T) / (dt * ndump))), 0):

        W = VectorFunctionSpace(mesh_, 'CG', 1) * FunctionSpace(mesh_, 'CG', 1)

        # Read in saved data from .h5:
        with DumbCheckpoint('data_dumps/tsunami/adjoint_soln_{y}'.format(y=i), mode=FILE_READ) as chk:
            lu_P1 = Function(W.sub(0), name='P1 adjoint velocity')
            le = Function(W.sub(1), name='Adjoint free surface')
            chk.load(lu_P1)
            chk.load(le)

        # Interpolate saved data onto new mesh:
        if i != i0:
            print '    #### Interpolation step', j - max(i, int((Ts - T) / (dt * ndump))) + 1, '/', \
                len(range(max(i, int((Ts - T) / (dt * ndump))), 0))
            lu_P1, le = interp(mesh, lu_P1, le)

        # Multiply fields together:
        ip.dat.data[:] = lu_P1.dat.data[:, 0] * vel.dat.data[:, 0] + lu_P1.dat.data[:, 1] * vel.dat.data[:, 1] \
                         + le.dat.data * eta.dat.data

        # Extract (pointwise) maximal values:
        if j == 0:
            significance.dat.data[:] = ip.dat.data[:]
        else:
            for k in range(len(ip.dat.data)):
                if np.abs(ip.dat.data[k]) > np.abs(significance.dat.data[k]):
                    significance.dat.data[k] = ip.dat.data[k]
    sig_file.write(significance, time=t)

    # Adapt mesh to significant data and interpolate:
    V = TensorFunctionSpace(mesh, 'CG', 1)
    H = construct_hessian(mesh, V, significance, method=hess_meth)
    M = compute_steady_metric(mesh, V, H, significance, h_min=hmin, h_max=hmax, normalise=ntype, num=nodes)
    adaptor = AnisotropicAdaptation(mesh, M)
    mesh = adaptor.adapted_mesh
    u, u_, eta, eta_, q, q_, b, W = interp_Taylor_Hood(mesh, u, u_, eta, eta_, b)
    vel = Function(VectorFunctionSpace(mesh, 'CG', 1))
    u.rename('Fluid velocity')
    eta.rename('Free surface displacement')
    i += 1

    # Mesh resolution analysis:
    n = len(mesh.coordinates.dat.data)
    if n < N1:
        N1 = n
    elif n > N2:
        N2 = n
    toc2 = clock()

    # Print to screen:
    print ''
    print '************ Adaption step %d **************' % (i + i0 + 1)
    print 'Time = %1.2f mins / %1.1f mins' % (t / 60., T / 60.)
    print 'Number of nodes after adaption', n
    print 'Min. nodes in mesh: %d... max. nodes in mesh: %d' % (N1, N2)
    print 'Total elapsed time for this step: %1.2fs' % (toc2 - tic2)
    print ''

    # Establish test functions and midpoint averages:
    v, ze = TestFunctions(W)
    u, eta = split(q)
    u_, eta_ = split(q_)
    uh = 0.5 * (u + u_)
    etah = 0.5 * (eta + eta_)

    # Set up the variational problem:
    Lf = (ze * (eta - eta_) - Dt * inner(b * uh, grad(ze)) + inner(u - u_, v) + Dt * g * (inner(grad(etah), v))) * dx
    q_prob = NonlinearVariationalProblem(Lf, q)
    q_solv = NonlinearVariationalSolver(q_prob, solver_parameters=params)

    # Split to access data and relabel functions:
    u, eta = q.split()
    u_, eta_ = q_.split()
    u.rename('Fluid velocity')
    eta.rename('Free surface displacement')

    # Inner timeloop:
    for k in range(rm):
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
print '\a'
toc1 = clock()
print 'Elapsed time for adaptive solver: %1.2fs' % (toc1 - tic1)

# Store gauge timeseries data to file:
gauge_timeseries(gauge, gauge_dat)