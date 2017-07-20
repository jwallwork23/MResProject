from firedrake import *
from firedrake_adjoint import *

import numpy as np
from time import clock

from utils import construct_hessian, compute_steady_metric, interp, interp_Taylor_Hood

# Define initial (uniform) mesh:
na = 8                                      # (Coarse) resolution of uniform mesh for adjoint run
lx = 4                                      # Extent in x-direction (m)
mesh_a = SquareMesh(lx * na, lx * na, lx, lx)

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

# Define mixed Taylor-Hood function space:
W_a = VectorFunctionSpace(mesh_a, 'CG', 2) * FunctionSpace(mesh_a, 'CG', 1)

# Create adjoint variables:
lam_ = Function(W_a)
lu_, le_ = lam_.split()

# Establish bathymetry functions on each mesh:
b_a = Function(W_a.sub(1), name='Coarse bathymetry')
b_a.interpolate(Expression('x[0] <= 0.5 ? 0.01 : 0.1'))     # Shelf break bathymetry at x = 50cm

# Establish indicator function for adjoint equations:
ind = Function(W_a.sub(1), name='Indicator function')
ind.interpolate(Expression('(x[0] >= 0.) & (x[0] < 0.3) & (x[1] > 1.5) & (x[1] < 2.5) ? 0.001 : 0'))

# Interpolate adjoint final time conditions:
lu_.interpolate(Expression([0, 0]))
le_.interpolate(Expression('(x[0] >= 0.) & (x[0] < 0.3) & (x[1] > 1.5) & (x[1] < 2.5) ? 0.001 : 0'))

# Set up dependent variables of the adjoint problem:
lam = Function(W_a)
lam.assign(lam_)
lu, le = lam.split()

# Label variables:
lu.rename('Adjoint velocity')
le.rename('Adjoint free surface')

# Initialise files:
lam_file = File('plots/adjoint_outputs/adjoint.pvd')
lam_file.write(lu, le, time=0)

# Establish test functions and midpoint averages:
w, xi = TestFunctions(W_a)
lu, le = split(lam)
lu_, le_ = split(lam_)
luh = 0.5 * (lu + lu_)
leh = 0.5 * (le + le_)

# Set up the variational problem:
La = ((le - le_) * xi - Dt * g * b_a * inner(luh, grad(xi)) #+ ind * xi
      + inner(lu - lu_, w) + Dt * b_a * inner(grad(leh), w)) * dx
lam_prob = NonlinearVariationalProblem(La, lam)
lam_solv = NonlinearVariationalSolver(lam_prob, solver_parameters={'mat_type': 'matfree',
                                                                   'snes_type': 'ksponly',
                                                                   'pc_type': 'python',
                                                                   'pc_python_type': 'firedrake.AssembledPC',
                                                                   'assembled_pc_type': 'lu',
                                                                   'snes_lag_preconditioner': -1,
                                                                   'snes_lag_preconditioner_persists': True})
# Split to access data and relabel functions:
lu, le = lam.split()
lu_, le_ = lam_.split()

# Initalise counters:
t = T
dumpn = ndump

# Run fixed mesh adjoint solver:
while t > 0.5 * dt:

    # Increment counters:
    t -= dt
    dumpn -= 1

    # Solve the problem and update:
    lam_solv.solve()
    lam_.assign(lam)

    if dumpn == 0:
        dumpn += ndump
        lam_file.write(lu, le, time=T-t)
        print 't = %1.2fs' % t

# [Fixed forward solver]

# Repeat above setup:
nf = 2 * na                                 # (Medium) resolution of uniform mesh for forward run
mesh_f = SquareMesh(lx * nf, lx * nf, lx, lx)
x, y = SpatialCoordinate(mesh_f)
W_f = VectorFunctionSpace(mesh_f, 'CG', 2) * FunctionSpace(mesh_f, 'CG', 1)
q_ = Function(W_f)
u_, eta_ = q_.split()
b_f = Function(W_f.sub(1), name='Fine bathymetry')
b_f.interpolate(Expression('x[0] <= 0.5 ? 0.01 : 0.1'))
# Interpolate forward initial conditions:
u_.interpolate(Expression([0, 0]))
eta_.interpolate(1e-3 * exp(- (pow(x - 2.5, 2) + pow(y - 2., 2)) / 0.04))
# Set up dependent variables of the forward problem:
q = Function(W_f)
q.assign(q_)
u, eta = q.split()
u.rename('Fluid velocity')
eta.rename('Free surface displacement')
q_file = File('plots/adjoint_outputs/forward.pvd')
q_file.write(u, eta, time=0)

# Initialise counters:
t = 0.
cnt = 0
dumpn = 0
mn = 0

# [Adaptive forward solver]