from firedrake import *

import numpy as np
from time import clock

from utils import adapt, construct_hessian, compute_steady_metric, interp, Meshd, metric_intersection, update_SW_FE

# Define initial (uniform) mesh:
n = int(raw_input('Mesh cells per m (default 16)?: ') or 16)            # Resolution of initial uniform mesh
lx = 4                                                                  # Extent in x-direction (m)                                                                          # Extent in y-direction (m)
mesh = SquareMesh(lx * n, lx * n, lx, lx)
meshd = Meshd(mesh)
x, y = SpatialCoordinate(mesh)
N1 = len(mesh.coordinates.dat.data)                                     # Minimum number of nodes
N2 = N1                                                                 # Maximum number of nodes
print 'Initial number of nodes : ', N1
bathy = raw_input('Flat bathymetry or shelf break (f/s)?: ') or 'f'
if (bathy != 'f') & (bathy != 's') :
    raise ValueError('Please try again, choosing f or s.')

# Simulation duration:
T = 2.5

# Set up adaptivity parameters:
remesh = raw_input('Use adaptive meshing (y/n)?: ') or 'y'
if remesh == 'y' :
    hmin = float(raw_input('Minimum element size in mm (default 5)?: ') or 5.) * 1e-3
    hmax = float(raw_input('Maximum element size in mm (default 100)?: ') or 100.) * 1e-3
    rm = int(raw_input('Timesteps per remesh (default 5)?: ') or 5)
    nodes = float(raw_input('Target number of nodes (default 1000)?: ') or 1000.)
    ntype = raw_input('Normalisation type? (lp/manual): ') or 'lp'
    mtype = raw_input('Mesh w.r.t. speed, free surface or both? (s/f/b): ') or 'f'
    if mtype not in ('s','f','b'):
        raise ValueError('Please try again, choosing s, f or b.')
else :
    hmin = 0.005
    rm = int(T)
    nodes = 0
    ntype = None
    mtype = None
    if remesh != 'n' :
        raise ValueError('Please try again, choosing y or n.')

# Define function spaces:
Vu = VectorFunctionSpace(mesh, 'CG', 1)                                     # TODO: consider Taylor-Hood elements
Ve = FunctionSpace(mesh, 'CG', 1)
Vq = MixedFunctionSpace((Vu, Ve))                                           # Mixed FE problem

# Establish bathymetry function:
b = Function(Ve, name = 'Bathymetry')
if bathy == 'f' :
    b.assign(0.1)  # (Constant) tank water depth (m)
else :
    b.interpolate(Expression('x[0] <= 0.5 ? 0.01 : 0.1'))  # Shelf break bathymetry

# Courant number adjusted timestepping parameters:
ndump = 1
g = 9.81                                    # Gravitational acceleration (m s^{-2})
dt = 0.8 * hmin / np.sqrt(g * 0.1)          # Timestep length (s), using wavespeed sqrt(gh)
Dt = Constant(dt)
print 'Using Courant number adjusted timestep dt = %1.4f' % dt

# Construct a function to store our two variables at time n:
q_ = Function(Vq)
u_, eta_ = q_.split()

# Interpolate initial conditions:
u_.interpolate(Expression([0, 0]))
eta_.interpolate(1e-3 * exp( - (pow(x - 2., 2) + pow(y - 2., 2)) / 0.04))

# Set up functions of forward weak problem:
q = Function(Vq)
q.assign(q_)
v, ze = TestFunctions(Vq)
u, eta = split(q)
u_, eta_ = split(q_)

# For timestepping we consider the implicit midpoint rule and so must create new 'mid-step' functions:
uh = 0.5 * (u + u_)
etah = 0.5 * (eta + eta_)

# Specify solver parameters:
params = {'mat_type': 'matfree',
          'snes_type': 'ksponly',
          'pc_type': 'python',
          'pc_python_type': 'firedrake.AssembledPC',
          'assembled_pc_type': 'lu',
          'snes_lag_preconditioner': -1,
          'snes_lag_preconditioner_persists': True,}

# Set up the variational problem:
L = (ze * (eta - eta_) - Dt * inner(b * uh, grad(ze)) + inner(u - u_, v) + Dt * g *(inner(grad(etah), v))) * dx
q_prob = NonlinearVariationalProblem(L, q)
q_solv = NonlinearVariationalSolver(q_prob, solver_parameters = params)

# 'Split' functions to access their data and relabel:
u_, eta_ = q_.split()
u, eta = q.split()
u.rename('Fluid velocity')
eta.rename('Free surface displacement')

# Initialise time, counters and files:
t = 0.
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

        if mtype != 'f' :

            # Establish velocity speed for adaption:
            spd = Function(FunctionSpace(mesh, 'CG', 1))
            spd.interpolate(sqrt(dot(u, u)))

            # Compute Hessian and metric:
            V = TensorFunctionSpace(mesh, 'CG', 1)
            H = construct_hessian(mesh, V, spd)
            M = compute_steady_metric(mesh, V, H, spd, h_min = hmin, h_max = hmax, N = nodes, normalise = ntype)
            M.rename('Metric field')

        if mtype != 's' :

            H = construct_hessian(mesh, V, eta)
            M2 = compute_steady_metric(mesh, V, H, eta, h_min = hmin, h_max = hmax, N = nodes, normalise = ntype)

            if mtype == 'b' :
                M = metric_intersection(mesh, V, M, M2)

            else :
                M = M2
            M.rename('Metric field')

            # Adapt mesh and set up new function spaces:
            mesh_ = mesh
            meshd_ = Meshd(mesh_)
            tic2 = clock()
            mesh = adapt(mesh, M)
            meshd = Meshd(mesh)
            q_, q, u_, u, eta_, eta, b, Vq = update_SW_FE(meshd_, meshd, u_, u, eta_, eta, b)
            toc2 = clock()

        # Data analysis:
        n = len(mesh.coordinates.dat.data)
        if n < N1 :
            N1 = n
        elif n > N2 :
            N2 = n

        # Print to screen:
        print ''
        print '************ Adaption step %d **************' % mn
        print 'Time = %1.2fs' % t
        print 'Number of nodes after adaption step %d: ' % mn, n
        print 'Min. nodes in mesh: %d... max. nodes in mesh: %d' % (N1, N2)
        print 'Elapsed time for adaption step %d: %1.2fs' % (mn, toc2 - tic2)
        print ''

    # Set up functions of weak problem:
    v, ze = TestFunctions(Vq)
    u, eta = split(q)
    u_, eta_ = split(q_)
    uh = 0.5 * (u + u_)
    etah = 0.5 * (eta + eta_)

    # Set up the variational problem:
    L = (ze * (eta - eta_) - Dt * inner(b * uh, grad(ze)) +
         inner(u - u_, v) + Dt * g * (inner(grad(etah), v))) * dx
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
                print 't = %1.2fs' % t

# End timing and print:
toc1 = clock()
if remesh == 'y' :
    print 'Elapsed time for adaptive solver: %1.2es' % (toc1 - tic1)
    print 'Minimum number of nodes: %1.4fs' % N1
    print 'Maximum number of nodes: %1.4fs' % N2
else :
    print 'Elapsed time for non-adaptive solver: %1.2es' % (toc1 - tic1)
