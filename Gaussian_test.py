from firedrake import *

import numpy as np
from time import clock

from utils import construct_hessian, compute_steady_metric, interp, interp_Taylor_Hood

# Define initial (uniform) mesh:
n = int(raw_input('Mesh cells per m (default 16)?: ') or 16)            # Resolution of initial uniform mesh
lx = 4                                                                  # Extent in x-direction (m)
mesh = SquareMesh(lx * n, lx * n, lx, lx)
x, y = SpatialCoordinate(mesh)
N1 = len(mesh.coordinates.dat.data)                                     # Minimum number of nodes
N2 = N1                                                                 # Maximum number of nodes
print 'Initial number of nodes : ', N1
bathy = raw_input('Flat bathymetry or shelf break (f/s)?: ') or 'f'
if bathy not in ('f', 's'):
    raise ValueError('Please try again, choosing f or s.')

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
g = 9.81                            # Gravitational acceleration (m s^{-2})
dt = 0.8 * hmin / np.sqrt(g * 0.1)  # Timestep length (s), using wavespeed sqrt(gh)
Dt = Constant(dt)
print 'Using Courant number adjusted timestep dt = %1.4f' % dt

# Define mixed Taylor-Hood function space and a function defined thereupon:
W = VectorFunctionSpace(mesh, 'CG', 2) * FunctionSpace(mesh, 'CG', 1)
q_ = Function(W)
u_, eta_ = q_.split()

# Establish bathymetry function:
b = Function(W.sub(1), name='Bathymetry')
if bathy == 'f':
    b.assign(0.1)  # (Constant) tank water depth (m)
else:
    b.interpolate(Expression('x[0] <= 0.5 ? 0.01 : 0.1'))  # Shelf break bathymetry

# Interpolate initial conditions:
u_.interpolate(Expression([0, 0]))
eta_.interpolate(1e-3 * exp(- (pow(x - 2., 2) + pow(y - 2., 2)) / 0.04))

# Set up dependent variables of problem:
q = Function(W)
q.assign(q_)
u, eta = q.split()

# # Establish exact solution function:
# sol = Function(Ve, name = 'Exact free surface')
# sol.interpolate(1e-3 * exp( - (pow(x - 2., 2) + pow(y - 2., 2)) / 0.04))
# k = 1.
# l = 1.
# kap = sqrt(pow(k, 2) + pow(l, 2))

# Initialise time, counters and files:
t = 0.
mn = 0
dumpn = 0
q_file = File('plots/adapt_plots/gaussian_test.pvd')
# ex_file = File('plots/adapt_plots/gaussian_exact.pvd')     TODO: Plot exact soln
# ex_file.write(sol, time = t)
m_file = File('plots/adapt_plots/gaussian_test_metric.pvd')
h_file = File('plots/adapt_plots/gaussian_test_hessian.pvd')
tic1 = clock()

# Enter timeloop:
while t < T - 0.5 * dt:

    # Update counters:
    mn += 1
    cnt = 0

    if remesh == 'y':

        # Compute Hessian and metric:
        V = TensorFunctionSpace(mesh, 'CG', 1)
        if mtype != 'f':
            spd = Function(FunctionSpace(mesh, 'CG', 1))        # Fluid speed
            spd.interpolate(sqrt(dot(u, u)))
            H = construct_hessian(mesh, V, spd, method=hess_meth)
            if mat_out == 'y':
                H.rename('Hessian')
                h_file.write(H, time=t)
            M = compute_steady_metric(mesh, V, H, spd, h_min=hmin, h_max=hmax, N=nodes, normalise=ntype)
        if mtype != 's':
            H = construct_hessian(mesh, V, eta, method=hess_meth)
            if (mtype != 'b') & (mat_out == 'y'):
                H.rename('Hessian')
                h_file.write(H, time=t)
            M2 = compute_steady_metric(mesh, V, H, eta, h_min=hmin, h_max=hmax, N=nodes, normalise=ntype)
            if mtype == 'b':
                M = metric_intersection(mesh, V, M, M2)
            else:
                M = M2
        if mat_out == 'y':
            M.rename('Metric field')
            m_file.write(M, time=t)

        # Adapt mesh and interpolate functions:
        tic2 = clock()
        adaptor = AnisotropicAdaptation(mesh, M)
        mesh = adaptor.adapted_mesh
        u_, eta_, q_, W = interp_Taylor_Hood(adaptor, u_, eta_)
        u, eta, q, W = interp_Taylor_Hood(adaptor, u, eta)
        b = interp(adaptor, b)
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

    # Label functions:
    u.rename('Fluid velocity')
    eta.rename('Free surface displacement')

    # Establish test functions and midpoint averages:
    v, ze = TestFunctions(W)
    u, eta = split(q)
    u_, eta_ = split(q_)
    uh = 0.5 * (u + u_)
    etah = 0.5 * (eta + eta_)

    # Set up the variational problem:
    L = (ze * (eta - eta_) - Dt * inner(b * uh, grad(ze)) + inner(u - u_, v) + Dt * g * (inner(grad(etah), v))) * dx
    q_prob = NonlinearVariationalProblem(L, q)
    q_solv = NonlinearVariationalSolver(q_prob, solver_parameters={'mat_type': 'matfree',
                                                                   'snes_type': 'ksponly',
                                                                   'pc_type': 'python',
                                                                   'pc_python_type': 'firedrake.AssembledPC',
                                                                   'assembled_pc_type': 'lu',
                                                                   'snes_lag_preconditioner': -1,
                                                                   'snes_lag_preconditioner_persists': True})
    # Split to access data:
    u, eta = q.split()
    u_, eta_ = q_.split()

    if t < 0.5 * dt:
        q_file.write(u, eta, time=0)

    # Enter inner timeloop:
    while cnt < rm:

        # Increment counters:
        t += dt
        cnt += 1
        dumpn += 1

        # Solve the problem and update:
        q_solv.solve()
        q_.assign(q)

        if dumpn == ndump:
            dumpn -= ndump
            q_file.write(u, eta, time=t)
            # sol.interpolate(1e-3 * cos( - kap * t * sqrt(g * 0.1)) * exp( - (pow(x - 2., 2) + pow(y - 2., 2)) / 0.04))
            # ex_file.write(sol, time = t)        # TODO: ^^ Implement properly. Why doesn't it work??
            if remesh == 'n':
                print 't = %1.2fs' % t

# End timing and print:
toc1 = clock()
if remesh == 'y':
    print 'Elapsed time for adaptive solver: %1.2fs' % (toc1 - tic1)
else:
    print 'Elapsed time for non-adaptive solver: %1.2fs' % (toc1 - tic1)
