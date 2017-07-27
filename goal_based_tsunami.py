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
print '******************************** GOAL-BASED ADAPTIVE TSUNAMI SIMULATION ********************************'
print ''
print 'Options...'

# Define initial mesh (courtesy of QMESH) and functions, with initial conditions set:
coarseness = int(raw_input('Mesh coarseness? (Integer in range 1-5, default 4): ') or 4)
mesh, W, q_, u_, eta_, lam_, lm_, le_, b = Tohoku_domain(coarseness)
N1 = len(mesh.coordinates.dat.data)                                     # Minimum number of vertices
N2 = N1                                                                 # Maximum number of vertices
print '...... mesh loaded. Initial number of vertices : ', N1

# Set physical parameters:
g = 9.81                        # Gravitational acceleration (m s^{-2})

# Simulation duration:
T = float(raw_input('Simulation duration in hours (default 1)?: ') or 1.) * 3600.

# # Set up adaptivity parameters:
# hmin = float(raw_input('Minimum element size in km (default 0.5)?: ') or 0.5) * 1e3
# hmax = float(raw_input('Maximum element size in km (default 10000)?: ') or 10000.) * 1e3
# rm = int(raw_input('Timesteps per re-mesh (default 10)?: ') or 10)
# ntype = raw_input('Normalisation type? (lp/manual): ') or 'lp'
# if ntype not in ('lp', 'manual'):
#     raise ValueError('Please try again, choosing lp or manual.')
# mtype = raw_input('Mesh w.r.t. speed, free surface or both? (s/f/b): ') or 's'
# if mtype not in ('s', 'f', 'b'):
#     raise ValueError('Please try again, choosing s, f or b.')
# hess_meth = raw_input('Integration by parts or double L2 projection? (parts/dL2): ') or 'dL2'
# if hess_meth not in ('parts', 'dL2'):
#     raise ValueError('Please try again, choosing parts or dL2.')
# nodes = 0.1 * N1    # Target number of vertices

# Courant number adjusted timestepping parameters:
dt = float(raw_input('Specify timestep in seconds (default 1): ') or 1.)
Dt = Constant(dt)
# cdt = hmin / np.sqrt(g * max(b.dat.data))
# if dt > cdt:
#     print 'WARNING: chosen timestep dt =', dt, 'exceeds recommended value of', cdt
#     if raw_input('Are you happy to proceed? (y/n)') == 'n':
#         exit(23)
# ndump = int(60. / dt)
ndump = 60

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
f.interpolate(Expression('(x[0] > 490e3) & (x[0] < 580e3) & (x[1] > 4130e3) & (x[1] < 4260e3) ? 1. : 0.'))

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
i = -1
dumpn = ndump

# Initialise tensor arrays for storage (with dimensions pre-allocated for speed):
velocity_dat = np.zeros((int(T / dt) + 1, N1, 2))
surface_dat = np.zeros((int(T / dt) + 1, N1))
inner_product_dat = np.zeros((int(T / dt) + 1, N1))
significant_dat = np.zeros(N1)
vel.interpolate(lu)
velocity_dat[i, :, :] = vel.dat.data
surface_dat[i, :] = le.dat.data

# Establish test functions and midpoint averages:
w, xi = TestFunctions(W)
lu, le = split(lam)
lu_, le_ = split(lam_)
luh = 0.5 * (lu + lu_)
leh = 0.5 * (le + le_)

# Set up the variational problem:
La = ((le - le_) * xi - Dt * g * b * inner(luh, grad(xi)) - f * xi
      + inner(lu - lu_, w) + Dt * b * inner(grad(leh), w)) * dx
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
    i -= 1
    dumpn -= 1

    # Solve the problem and update:
    lam_solv.solve()
    lam_.assign(lam)

    # Save data:
    vel.interpolate(lu)
    velocity_dat[i, :, :] = vel.dat.data
    surface_dat[i, :] = le.dat.data

    # Dump to vtu:
    if dumpn == 0:
        dumpn += ndump
        lam_file.write(lu, le, time=T-t)
        print 't = %1.2f mins' % (t / 60.)
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

# Initialise counters:
t = 0.
dumpn = 0

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

# Create a function to hold the inner product data:
ip = Function(W.sub(1), name='Inner product')
ip.interpolate(Expression(0))
ip_file = File('plots/goal-based_outputs/tsunami_inner_product.pvd')
ip_file.write(ip, time=t)

print ''
print 'Starting fixed resolution forward run...'
while t < T - 0.5 * dt:

    # Increment counters:
    t += dt
    i += 1
    dumpn += 1

    # Solve the problem and update:
    q_solv.solve()
    q_.assign(q)

    # Take inner product with adjoint data:
    vel.interpolate(u)
    velocity_dat[i, :, :] = velocity_dat[i, :, :] * vel.dat.data
    surface_dat[i, :] = surface_dat[i, :] * eta.dat.data
    inner_product_dat[i, :] = velocity_dat[i, :, 0] + velocity_dat[i, :, 1] + surface_dat[i, :]

    # Take maximum as most significant:
    for j in range(N1):
        if np.abs(inner_product_dat[i, j]) > significant_dat[j]:
            significant_dat[j] = np.abs(inner_product_dat[i, j])

    # Dump to vtu:
    if dumpn == ndump:
        dumpn -= ndump
        q_file.write(u, eta, time=t)
        ip.dat.data[:] = inner_product_dat[i, :]
        ip_file.write(ip, time=t)
        print 't = %1.2f mins' % (t / 60.)
print '... done!'

significance = Function(W.sub(1), name='Significant regions')
significance.dat.data[:] = significant_dat[:]
File('plots/goal-based_outputs/tsunami_significance.pvd').write(significance)