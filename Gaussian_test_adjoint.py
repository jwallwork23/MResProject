from firedrake import *
from firedrake_adjoint import *

import numpy as np
from time import clock

from utils import construct_hessian, compute_steady_metric, interp, interp_Taylor_Hood

# Define initial (uniform) mesh:
n_adj = 8                                   # (Coarse) resolution of uniform mesh for adjoint run
n_for = 2 * n_adj                           # (Medium) resolution of uniform mesh for forward run
lx = 4                                      # Extent in x-direction (m)
mesh_f = SquareMesh(lx * n_for, lx * n_for, lx, lx)
mesh_a = SquareMesh(lx * n_adj, lx * n_adj, lx, lx)

# Simulation duration:
T = 2.5

# Set up adaptivity parameters:
remesh = raw_input('Use adaptive meshing (y/n)?: ') or 'y'
if remesh == 'y':
    hmin = float(raw_input('Minimum element size in mm (default 5)?: ') or 5.) * 1e-3
    hmax = float(raw_input('Maximum element size in mm (default 100)?: ') or 100.) * 1e-3
    rm = int(raw_input('Timesteps per re-mesh (default 5)?: ') or 5)
    nodes = float(raw_input('Target number of nodes (default 1000)?: ') or 1000.)
    ntype = raw_input('Normalisation type? (lp/manual): ') or 'lp'
    if ntype not in ('lp', 'manual'):
        raise ValueError('Please try again, choosing lp or manual.')
    mtype = raw_input('Mesh w.r.t. speed, free surface or both? (s/f/b): ') or 'f'
    if mtype not in ('s', 'f', 'b'):
        raise ValueError('Please try again, choosing s, f or b.')
    mat_out = raw_input('Output Hessian and metric? (y/n): ') or 'n'
    if mat_out not in ('y', 'n'):
        raise ValueError('Please try again, choosing y or n.')
    hess_meth = raw_input('Integration by parts or double L2 projection? (parts/dL2): ') or 'dL2'
    if hess_meth not in ('parts', 'dL2'):
        raise ValueError('Please try again, choosing parts or dL2.')
else:
    hmin = 0.0625
    rm = int(T)
    nodes = 0
    ntype = None
    mtype = None
    mat_out = 'n'
    if remesh != 'n':
        raise ValueError('Please try again, choosing y or n.')

# Courant number adjusted timestepping parameters:
ndump = 1
g = 9.81                                                # Gravitational acceleration (m s^{-2})
dt = 0.8 * hmin / np.sqrt(g * 0.1)                      # Timestep length (s), using wavespeed sqrt(gh)
Dt = Constant(dt)
print 'Using Courant number adjusted timestep dt = %1.4f' % dt

# Define mixed Taylor-Hood function space:
W_f = VectorFunctionSpace(mesh_f, 'CG', 2) * FunctionSpace(mesh_f, 'CG', 1)
W_a = VectorFunctionSpace(mesh_a, 'CG', 2) * FunctionSpace(mesh_a, 'CG', 1)

# Create forward variables:
q_ = Function(W_f)
u_, eta_ = q_.split()

# Create adjoint variables:
lam_ = Function(W_a)
lu_, le_ = lam_.split()

# Establish bathymetry functions on each mesh:
b_f = Function(W_f.sub(1), name='Bathymetry')
b_f.interpolate(Expression('x[0] <= 0.5 ? 0.01 : 0.1'))   # Shelf break bathymetry at x = 50cm
b_a = Function(W_a.sub(1), name='Bathymetry')
b_a.interpolate(Expression('x[0] <= 0.5 ? 0.01 : 0.1'))

# Interpolate forward initial conditions:
u_.interpolate(Expression([0, 0]))
eta_.interpolate(1e-3 * exp(- (pow(x - 2.5, 2) + pow(y - 2., 2)) / 0.04))

# Establish indicator function for adjoint equations:
ind = Function(W_a.sub(1), name='Indicator function')
le_.interpolate(Expression('(x[0] > 0.1) & (x[0] < 0.4) & (x[1] > 1.5) & x[1] < 2.5) ? 1 : 0'))

# Interpolate adjoint final time conditions:
lu_.interpolate(Expression([0, 0]))
le_.assign(ind)

# Set up dependent variables of the forward problem:
q = Function(W)
q.assign(q_)
u, eta = q.split()

# Set up dependent variables of the adjoint problem:
lam = Function(W)
lam.assign(lam_)
lu, le = lam.split()

# Label variables:
u.rename('Fluid velocity')
eta.rename('Free surface displacement')
lu.rename('Adjoint velocity')
le.rename('Adjoint free surface')

# Initialise files:
q_file = File('plots/adjoint_outputs/forward.pvd')
q_file.write(u, eta, time=0)
lam_file = File('plots/adjoint_outputs/adjoint.pvd')
lam_file.write(lu, le, time=0)

# [Fixed adjoint solver]

# [Fixed forward solver]

# Initialise counters:
t = 0.
cnt = 0
dumpn = 0
mn = 0

# [Adaptive forward solver]