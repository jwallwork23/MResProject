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
tic1 = clock()
print 'Options...'

# Define initial mesh (courtesy of QMESH) and functions, with initial conditions set:
coarseness = int(raw_input('Mesh coarseness? (Integer in range 1-5, default 5): ') or 5)
mesh, W, q_, u_, eta_, lam_, lm_, le_, b = Tohoku_domain(coarseness)
mesh_ = mesh
N1 = len(mesh.coordinates.dat.data)                                     # Minimum number of vertices
N2 = N1                                                                 # Maximum number of vertices
print '...... mesh loaded. Initial number of vertices : ', N1

# Set up adaptivity parameters:
hmin = float(raw_input('Minimum element size in km (default 0.5)?: ') or 0.5) * 1e3
hmax = float(raw_input('Maximum element size in km (default 10000)?: ') or 10000.) * 1e3
rm = int(raw_input('Timesteps per re-mesh (default 60)?: ') or 60)
ntype = raw_input('Normalisation type? (lp/manual): ') or 'lp'
if ntype not in ('lp', 'manual'):
    raise ValueError('Please try again, choosing lp or manual.')
mtype = raw_input('Mesh w.r.t. speed, free surface or both? (s/f/b, default f): ') or 'f'
if mtype not in ('s', 'f', 'b'):
    raise ValueError('Please try again, choosing s, f or b.')
hess_meth = raw_input('Integration by parts or double L2 projection? (parts/dL2, default dL2): ') or 'dL2'
if hess_meth not in ('parts', 'dL2'):
    raise ValueError('Please try again, choosing parts or dL2.')
nodes = 0.5 * N1                # Target number of vertices

# Specify parameters:
ndump = 30                      # Timesteps per data dump
T = float(raw_input('Simulation duration in hours (default 0.415)?: ') or 0.415) * 3600.
Ts = 300.                       # Time range lower limit (s), during which we can assume the wave won't reach the shore
g = 9.81                        # Gravitational acceleration (m s^{-2})
dt = float(raw_input('Specify timestep in seconds (default 1): ') or 1.)
Dt = Constant(dt)
rm = 60                         # Timesteps per remesh

# Specify solver parameters:
params = {'mat_type': 'matfree',
          'snes_type': 'ksponly',
          'pc_type': 'python',
          'pc_python_type': 'firedrake.AssembledPC',
          'assembled_pc_type': 'lu',
          'snes_lag_preconditioner': -1,
          'snes_lag_preconditioner_persists': True}

# Establish indicator function for adjoint equations:
f = Function(W.sub(1), name='Forcing term')
f.interpolate(Expression('(x[0] > 490e3) & (x[0] < 580e3) & (x[1] > 4130e3) & (x[1] < 4260e3) ? 20. : 0.'))

# Set up dependent variables of the adjoint problem:
lam = Function(W)
lam.assign(lam_)
lu, le = lam.split()
vel = Function(VectorFunctionSpace(mesh, 'CG', 1))          # For interpolating velocity field

# Label variables:
lu.rename('Adjoint velocity')
le.rename('Adjoint free surface')

# Initialise files:
lam_file = File('plots/goal-based_outputs/tsunami_adjoint.pvd')
lam_file.write(lu, le, time=0)

# Initalise counters:
t = T
i = 0
dumpn = ndump

# Initialise tensor arrays for storage (with dimensions pre-allocated for speed):
sol_dat = np.zeros((int(T / (dt * ndump)) + 1, N1, 3))
significant_dat = np.zeros((int(T / (dt * ndump)) + 1, N1))
vel.interpolate(lu)
sol_dat[i, :, :2] = vel.dat.data
sol_dat[i, :, 2] = le.dat.data

# Establish test functions and midpoint averages:
w, xi = TestFunctions(W)
lu, le = split(lam)
lu_, le_ = split(lam_)
luh = 0.5 * (lu + lu_)
leh = 0.5 * (le + le_)

# Set up the variational problem:
La = ((le - le_) * xi - Dt * g * b * inner(luh, grad(xi)) + f * xi
      + inner(lu - lu_, w) + Dt * (b * inner(grad(leh), w) + leh * inner(grad(b), w))) * dx
lam_prob = NonlinearVariationalProblem(La, lam)
lam_solv = NonlinearVariationalSolver(lam_prob, solver_parameters=params)

# Split to access data and relabel functions:
lu, le = lam.split()
lu_, le_ = lam_.split()

print ''
print 'Starting fixed resolution adjoint run...'
while t > 0.5 * dt:

    # Increment counters:
    t -= dt
    dumpn -= 1

    # Solve the problem and update:
    lam_solv.solve()
    lam_.assign(lam)

    # Dump to vtu:
    if dumpn == 0:
        i -= 1
        dumpn += ndump

        # Save data:
        vel.interpolate(lu)
        sol_dat[i, :, :2] = vel.dat.data
        sol_dat[i, :, 2] = le.dat.data

        lam_file.write(lu, le, time=T-t)
        print 't = %1.1f mins' % (t / 60.)
print '... done!'

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

# Initialise counters:
t = 0.
dumpn = 0

print ''
print 'Starting mesh adaptive forward run...'
while t < T - 0.5 * dt:
    tic2 = clock()
    i += 1

    # Interpolate velocity in a P1 space:
    vel.interpolate(u)

    # Create functions to hold inner product and significance data:
    ip = Function(W.sub(1), name='Inner product')
    significance = Function(W.sub(1), name='Significant regions')

    # Take maximal L2 inner product as most significant:
    for j in range(max(i, int((Ts - T) / (dt * ndump))), 0):

        sol_v = Function(VectorFunctionSpace(mesh_, 'CG', 1))
        sol_e = Function(FunctionSpace(mesh_, 'CG', 1))
        sol_v.dat.data[:, :] = sol_dat[j, :, :2]
        sol_e.dat.data[:] = sol_dat[j, :, 2]

        # Interpolate saved data onto new mesh:
        if (i + int(T / (dt * ndump))) != 1:
            print '#### Interpolation step', j - max(i, int((Ts - T) / (dt * ndump))) + 1, '/', \
                len(range(max(i, int((Ts - T) / (dt * ndump))), 0))
            sol_v, sol_e = interp(mesh, sol_v, sol_e)

        ip.dat.data[:] = sol_v.dat.data[:, 0] * vel.dat.data[:, 0] + sol_v.dat.data[:, 1] * vel.dat.data[:, 1] \
                         + sol_e.dat.data * eta.dat.data
        if (j == 0) | (np.abs(assemble(ip * dx)) > np.abs(assemble(significance * dx))):
            significance.dat.data[:] = ip.dat.data[:]

    # Adapt mesh to significant data and interpolate:
    V = TensorFunctionSpace(mesh, 'CG', 1)
    H = construct_hessian(mesh, V, significance)
    M = compute_steady_metric(mesh, V, H, significance, h_min=0.1, h_max=5)
    adaptor = AnisotropicAdaptation(mesh, M)
    mesh = adaptor.adapted_mesh
    u, u_, eta, eta_, q, q_, b, W = interp_Taylor_Hood(mesh, u, u_, eta, eta_, b)
    vel = Function(VectorFunctionSpace(mesh, 'CG', 1))
    u.rename('Fluid velocity')
    eta.rename('Free surface displacement')

    # Data analysis:
    n = len(mesh.coordinates.dat.data)
    if n < N1:
        N1 = n
    elif n > N2:
        N2 = n
    toc2 = clock()

    # Print to screen:
    print ''
    print '************ Adaption step %d **************' % (i + 40)
    print 'Time = %1.1fs' % t
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

        # Dump to vtu:
        if dumpn == ndump:
            dumpn -= ndump
            q_file.write(u, eta, time=t)
            sig_file.write(significance, time=t)

toc1 = clock()
print 'Elapsed time for adaptive solver: %1.2f mins' % ((toc1 - tic1) / 60.)
