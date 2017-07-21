from firedrake import *
# from firedrake_adjoint import *

import numpy as np
from time import clock

from utils import construct_hessian, compute_steady_metric, interp, interp_Taylor_Hood

# Define initial (uniform) mesh:
n = 16                                      # Resolution of uniform mesh for adjoint run
lx = 4                                      # Extent in x-direction (m)
nx = n * lx
mesh = SquareMesh(nx, nx, lx, lx)
x, y = SpatialCoordinate(mesh)

# Simulation duration:
T = 20.

# Set up adaptivity parameters:
hmin = float(raw_input('Minimum element size in mm (default 5)?: ') or 5.) * 1e-3
# hmax = float(raw_input('Maximum element size in mm (default 100)?: ') or 100.) * 1e-3
# rm = int(raw_input('Timesteps per re-mesh (default 5)?: ') or 5)
# nodes = float(raw_input('Target number of nodes (default 1000)?: ') or 1000.)
# ntype = raw_input('Normalisation type? (lp/manual): ') or 'lp'
# if ntype not in ('lp', 'manual'):
#     raise ValueError('Please try again, choosing lp or manual.')
# mtype = raw_input('Mesh w.r.t. speed, free surface or both? (s/f/b): ') or 'f'
# if mtype not in ('s', 'f', 'b'):
#     raise ValueError('Please try again, choosing s, f or b.')
# mat_out = raw_input('Output Hessian and metric? (y/n): ') or 'n'
# if mat_out not in ('y', 'n'):
#     raise ValueError('Please try again, choosing y or n.')
# hess_meth = raw_input('Integration by parts or double L2 projection? (parts/dL2): ') or 'dL2'
# if hess_meth not in ('parts', 'dL2'):
#     raise ValueError('Please try again, choosing parts or dL2.')

# Courant number adjusted timestepping parameters:
ndump = 20
g = 9.81                                                # Gravitational acceleration (m s^{-2})
dt = 0.8 * hmin / np.sqrt(g * 0.1)                      # Timestep length (s), using wavespeed sqrt(gh)
Dt = Constant(dt)
print 'Using Courant number adjusted timestep dt = %1.4f' % dt

# Specify solver parameters:
params = {'mat_type': 'matfree',
          'snes_type': 'ksponly',
          'pc_type': 'python',
          'pc_python_type': 'firedrake.AssembledPC',
          'assembled_pc_type': 'lu',
          'snes_lag_preconditioner': -1,
          'snes_lag_preconditioner_persists': True}

# Define mixed Taylor-Hood function space:
W = VectorFunctionSpace(mesh, 'CG', 2) * FunctionSpace(mesh, 'CG', 1)

# Create adjoint variables:
lam_ = Function(W)
lu_, le_ = lam_.split()

# # Establish bathymetry functions on each mesh:
# b = Function(W_a.sub(1), name='Coarse bathymetry')
# b.interpolate(Expression('x[0] <= 0.5 ? 0.01 : 0.1'))     # Shelf break bathymetry at x = 50cm
b = 0.1

# Establish indicator function for adjoint equations:
ind = Function(W.sub(1), name='Indicator function')
ind.interpolate(Expression('(x[0] >= 0.) & (x[0] < 0.3) & (x[1] > 1.5) & (x[1] < 2.5) ? 0.001 : 0'))

# Interpolate adjoint final time conditions:
lu_.interpolate(Expression([0, 0]))
le_.assign(ind)

# Set up dependent variables of the adjoint problem:
lam = Function(W)
lam.assign(lam_)
lu, le = lam.split()

# Label variables:
lu.rename('Adjoint velocity')
le.rename('Adjoint free surface')

# Initialise files:
lam_file = File('plots/adjoint_outputs/adjoint.pvd')
lam_file.write(lu, le, time=0)

# Initalise counters:
t = T
i = -1
dumpn = ndump

# Initialise tensor arrays for storage (with dimensions pre-allocated for speed):
velocity_dat = np.zeros((int(T / (ndump * dt)) + 1, (2 * nx + 1) ** 2, 2))
surface_dat = np.zeros((int(T / (ndump * dt)) + 1, (nx + 1) ** 2 + 2))
significant_dat = np.zeros(((nx + 1) ** 2 + 2))
velocity_dat[i, :, :] = lm.dat.data
surface_dat[i, :] = le.dat.data

# Establish test functions and midpoint averages:
w, xi = TestFunctions(W)
lu, le = split(lam)
lu_, le_ = split(lam_)
luh = 0.5 * (lu + lu_)
leh = 0.5 * (le + le_)

# Set up the variational problem:
La = ((le - le_) * xi - Dt * g * b * inner(luh, grad(xi)) + inner(lu - lu_, w) + Dt * b * inner(grad(leh), w)) * dx
# + ind * xi * dx
lam_prob = NonlinearVariationalProblem(La, lam)
lam_solv = NonlinearVariationalSolver(lam_prob, solver_parameters=params)

# Split to access data and relabel functions:
lu, le = lam.split()
lu_, le_ = lam_.split()

# Run fixed mesh adjoint solver:
while t > 0.5 * dt:

    # Increment counters:
    t -= dt
    i -= 1
    dumpn -= 1

    # Solve the problem and update:
    lam_solv.solve()
    lam_.assign(lam)

    # Save data:
    velocity_dat[-i, :, :] = lm.dat.data
    surface_dat[-i, :] = le.dat.data

    # Dump to vtu:
    if dumpn == 0:
        dumpn += ndump
        lam_file.write(lu, le, time=T-t)
        print 't = %1.2fs' % t

# Repeat above setup:
q_ = Function(W)
u_, eta_ = q_.split()

# Interpolate forward initial conditions:
u_.interpolate(Expression([0, 0]))
eta_.interpolate(1e-3 * exp(- (pow(x - 2.5, 2) + pow(y - 2., 2)) / 0.04))

# Set up dependent variables of the forward problem:
q = Function(W)
q.assign(q_)
u, eta = q.split()

# Label variables:
u.rename('Fluid velocity')
eta.rename('Free surface displacement')

# Intialise files:
q_file = File('plots/adjoint_outputs/forward.pvd')
q_file.write(u, eta, time=0)

# Initialise counters:
t = 0.
i = 0
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

# Run fixed mesh forward solver:
while t < T - 0.5 * dt:

    # Increment counters:
    t += dt
    i += 1
    dumpn += 1

    # Solve the problem and update:
    q_solv.solve()
    q_.assign(q)

    # Take inner product with adjoint data:
    velocity_dat[i, :, 0] = velocity_dat[i, :, 0] * u.dat.data[:, :, 0]
    velocity_dat[i, :, 1] = velocity_dat[i, :, 1] * u.dat.data[:, :, 1]
    surface_dat[i, :] = surface_dat[i, :] * eta.dat.data

    # Dump to vtu:
    if dumpn == ndump:
        dumpn -= ndump
        q_file.write(u, eta, time=t)
        print 't = %1.2fs' % t

# [Adaptive forward solver]
