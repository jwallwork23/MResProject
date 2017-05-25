from firedrake import *

import numpy as np
from time import clock

from utils import adapt, construct_hessian, compute_steady_metric, interp, Meshd, update_tank_SW

# Define initial (uniform) mesh:
n = int(raw_input('Mesh cells per m (default 16)?: ') or 16)                    # Resolution of initial uniform mesh
lx = 4                                                                          # Extent in x-direction (m)                                                                          # Extent in y-direction (m)
mesh = SquareMesh(lx * n, lx * n, lx, lx)
meshd = Meshd(mesh)
x, y = SpatialCoordinate(mesh)
print 'Initial number of nodes : ', len(mesh.coordinates.dat.data)
b = Constant(0.1)       # (Constant) tank water depth (m)

# Specify timestepping parameters:
ndump = int(raw_input('Timesteps per data dump (default 1): ') or 1)
T = 2.5                                                                         # Simulation end time (s)
dt = 0.1/(n * ndump)                                                            # Timestep length (s)
Dt = Constant(dt)

# Set up adaptivity parameters:
remesh = raw_input('Use adaptive meshing (y/n)?: ') or 'y'
if remesh == 'y' :
    hmin = float(raw_input('Minimum element size in mm (default 5)?: ') or 5.) * 1e-3
    hmax = float(raw_input('Maximum element size in mm (default 100)?: ') or 100.) * 1e-3
    rm = int(raw_input('Timesteps per remesh (default 5)?: ') or 5)
    nodes = float(raw_input('Target number of nodes (default 1000)?: ') or 1000.)
    ntype = raw_input('Normalisation type? (lp/manual): ') or 'lp'
else :
    hmin = 0
    rm = int(T / dt)
    nodes = 0
    ntype = None

# Define function spaces:
Vu = VectorFunctionSpace(mesh, 'CG', 1)                                     # TODO: consider Taylor-Hood elements
Ve = FunctionSpace(mesh, 'CG', 1)
Vq = MixedFunctionSpace((Vu, Ve))                                           # Mixed FE problem

# Construct a function to store our two variables at time n:
q_ = Function(Vq)                                                           # Forward solution tuple
u_, eta_ = q_.split()

# Interpolate initial conditions:
u_.interpolate(Expression([0, 0]))
eta_.interpolate(1e-3 * exp(-(pow(x - 2., 2) + pow(y - 2., 2)) / 0.04))

# Set up functions of forward weak problem:
q = Function(Vq)
q.assign(q_)
v, ze = TestFunctions(Vq)
u, eta = split(q)
u_, eta_ = split(q_)

# Specify solver parameters:
params = {'mat_type': 'matfree',
          'snes_type': 'ksponly',
          'pc_type': 'python',
          'pc_python_type': 'firedrake.AssembledPC',
          'assembled_pc_type': 'lu',
          'snes_lag_preconditioner': -1,
          'snes_lag_preconditioner_persists': True,}

# Set up the variational problem:
g = 9.81            # Gravitational acceleration (m s^{-2})
L = (ze * (eta - eta_) - Dt * inner((eta + b) * u, grad(ze)) + inner(u - u_, v) + Dt * g *(inner(grad(eta), v))) * dx
q_prob = NonlinearVariationalProblem(L, q)
q_solv = NonlinearVariationalSolver(q_prob, solver_parameters = params)

# 'Split' functions to access their data and relabel:
u_, eta_ = q_.split()
u, eta = q.split()
u.rename('Fluid velocity')
eta.rename('Free surface displacement')

# Initialise time, counters and files:
t = 0.0
dumpn = 0
mn = 0
cnt = 0
i = 0
q_file = File('plots/adapt_plots/gaussian_test.pvd')
m_file = File('plots/adapt_plots/advection_test_metric.pvd')
q_file.write(u, eta, time = t)
tic1 = clock()

# Enter timeloop:
while t < T - 0.5 * dt :

    # Update counters:
    mn += 1
    cnt = 0

    if remesh == 'y' :

        # Compute Hessian and metric:
        V = TensorFunctionSpace(mesh, 'CG', 1)
        H = construct_hessian(mesh, V, eta)
        M = compute_steady_metric(mesh, V, H, eta, h_min = hmin, h_max = hmax, N = nodes)
        M.rename('Metric field')

        # Adapt mesh and set up new function spaces:
        mesh_ = mesh
        meshd_ = Meshd(mesh_)
        tic2 = clock()
        mesh = adapt(mesh, M)
        meshd = Meshd(mesh)
        q_, q, u_, u, eta_, eta, Vq = update_tank_SW(meshd_, meshd, u_, u, eta_, eta)
        toc2 = clock()

        # Print to screen:
        print ''
        print '************ Adaption step %d **************' % mn
        print 'Time = %1.2fs' % t
        print 'Number of nodes after adaption step %d: ' % mn, len(mesh.coordinates.dat.data)
        print 'Elapsed time for adaption step %d: %1.2es' % (mn, toc2 - tic2)
        print ''

    # Set up functions of weak problem:
    v, ze = TestFunctions(Vq)
    u, eta = split(q)
    u_, eta_ = split(q_)

    # Set up the variational problem:
    L = (ze * (eta - eta_) - Dt * inner((eta + b) * u, grad(ze)) +
         inner(u - u_, v) + Dt * g * (inner(grad(eta), v))) * dx
    q_prob = NonlinearVariationalProblem(L, q)
    q_solv = NonlinearVariationalSolver(q_prob, solver_parameters = params)

    # 'Split' functions to access their data and relabel:
    u_, eta_ = q_.split()
    u, eta = q.split()
    u.rename('Fluid velocity')
    eta.rename('Free surface displacement')

    # Enter inner timeloop:
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
                print 't = %1.2fs, mesh number =', t

# End timing and print:
toc1 = clock()
if remesh == 'y' :
    print 'Elapsed time for adaptive solver: %1.2es' % (toc1 - tic1)
else :
    print 'Elapsed time for non-adaptive solver: %1.2es' % (toc1 - tic1)
