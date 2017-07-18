from firedrake import *

import numpy as np
from time import clock

from utils import construct_hessian, compute_steady_metric, interp, Meshd, relab

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
if bathy not in ('f', 's') :
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
    if ntype not in ('lp', 'manual') :
        raise ValueError('Please try again, choosing lp or manual.')
    mtype = raw_input('Mesh w.r.t. speed, free surface or both? (s/f/b): ') or 'f'
    if mtype not in ('s', 'f', 'b'):
        raise ValueError('Please try again, choosing s, f or b.')
    mat_out = raw_input('Output Hessian and metric? (y/n): ') or 'n'
    if mat_out not in ('y', 'n') :
        raise ValueError('Please try again, choosing y or n.')
else :
    hmin = 0.0625
    rm = int(T)
    nodes = 0
    ntype = None
    mtype = None
    mat_out = 'n'
    if remesh != 'n' :
        raise ValueError('Please try again, choosing y or n.')

# Courant number adjusted timestepping parameters:
ndump = 1
g = 9.81  # Gravitational acceleration (m s^{-2})
dt = 0.8 * hmin / np.sqrt(g * 0.1)  # Timestep length (s), using wavespeed sqrt(gh)
Dt = Constant(dt)
print 'Using Courant number adjusted timestep dt = %1.4f' % dt

# Define mixed Taylor-Hood function space:
W = MixedFunctionSpace((VectorFunctionSpace(mesh, 'CG', 2), FunctionSpace(mesh, 'CG', 1)))                                           # Mixed FE problem

# Establish bathymetry function:
b = Function(W.sub(1), name = 'Bathymetry')
if bathy == 'f' :
    b.assign(0.1)  # (Constant) tank water depth (m)
else :
    b.interpolate(Expression('x[0] <= 0.5 ? 0.01 : 0.1'))  # Shelf break bathymetry

# Construct functions to store our variables at the previous timestep:
u_ = Function(W.sub(0), name='Fluid velocity')
eta_ = Function(W.sub(1), name='Free surface displacement')

# Interpolate initial conditions:
u_.interpolate(Expression([0, 0]))
eta_.interpolate(1e-3 * exp( - (pow(x - 2., 2) + pow(y - 2., 2)) / 0.04))

# # Establish exact solution function:
# sol = Function(Ve, name = 'Exact free surface')
# sol.interpolate(1e-3 * exp( - (pow(x - 2., 2) + pow(y - 2., 2)) / 0.04))
# k = 1.
# l = 1.
# kap = sqrt(pow(k, 2) + pow(l, 2))

# Set up functions of forward weak problem:
u, eta = TrialFunctions(W)
v, ze = TestFunctions(W)

# Initialise time, counters and files:
t = 0.
dumpn = 0
mn = 0
cnt = 0
i = 0
q_file = File('plots/adapt_plots/gaussian_test.pvd')
q_file.write(u_, eta_, time = t)
#ex_file = File('plots/adapt_plots/gaussian_exact.pvd')     TODO: Plot exact soln
#ex_file.write(sol, time = t)
if mat_out == 'y' :
    m_file = File('plots/adapt_plots/gaussian_test_metric.pvd')
    h_file = File('plots/adapt_plots/gaussian_test_hessian.pvd')
tic1 = clock()

# Enter timeloop:
while t < T - 0.5 * dt :

    # Update counters:
    mn += 1
    cnt = 0

    if remesh == 'y' :

        V = TensorFunctionSpace(mesh, 'CG', 1)

        if mtype != 'f' :

            # Establish fluid speed for adaption:
            spd = Function(FunctionSpace(mesh, 'CG', 1))
            spd.interpolate(sqrt(dot(u, u)))

            # Compute Hessian and metric:
            H = construct_hessian(mesh, V, spd)
            if mat_out == 'y' :
                H.rename('Hessian')
                h_file.write(H, time = t)
            M = compute_steady_metric(mesh, V, H, spd, h_min = hmin, h_max = hmax, N = nodes, normalise = ntype)

        if mtype != 's' :

            # Compute Hessian and metric:
            H = construct_hessian(mesh, V, eta)

            if (mtype != 'b') & (mat_out == 'y') :
                H.rename('Hessian')
                h_file.write(H, time = t)

            M2 = compute_steady_metric(mesh, V, H, eta, h_min = hmin, h_max = hmax, N = nodes, normalise = ntype)

            if mtype == 'b' :
                M = metric_intersection(mesh, V, M, M2)

            else :
                M = M2

        # Adapt mesh and interpolate functions:
        tic2 = clock()
        adaptor = AnisotropicAdaptation(mesh, M)
        mesh = adaptor.adapted_mesh
        meshd = Meshd(mesh)

        u_2, u2, eta_2, eta2, b2 = relab(u_, u, eta_, eta, b)

        # Re-define mixed Taylor-Hood function space:
        W = MixedFunctionSpace((VectorFunctionSpace(mesh, 'CG', 2), FunctionSpace(mesh, 'CG', 1)))

        # TODO: needs changing
        q_ = Function(Vq)
        u_, eta_ = q_.split()
        q = Function(Vq)
        u, eta = q.split()

        # Interpolate functions across from the previous mesh:
        u_, u, eta_, eta, b = interp(adaptor, u_2, u2, eta_2, eta2, b2)


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
        print 'Elapsed time for this step: %1.2fs' % (toc2 - tic2)
        print ''

    # Enter inner timeloop:
    while cnt < rm :

        t += dt
        cnt += 1
        dumpn += 1

        # Set up the forms of the variational problem:
        a = (ze * eta - 0.5 * Dt * inner(b * u, grad(ze)) + inner(u, v) + 0.5 * Dt * g * (inner(grad(eta), v))) * dx
        L = (ze * eta_ + 0.5 * Dt * inner(b * u_, grad(ze)) - inner(u_, v) - 0.5 * Dt * g * (inner(grad(eta_), v))) * dx
        q = Function(W)

        # Solve the problem:
        solve(a == L, q, solver_parameters={'mat_type': 'matfree',
                                            'snes_type': 'ksponly',
                                            'pc_type': 'python',
                                            'pc_python_type': 'firedrake.AssembledPC',
                                            'assembled_pc_type': 'lu',
                                            'snes_lag_preconditioner': -1,
                                            'snes_lag_preconditioner_persists': True,})
        u, eta = q.split()
        u.rename('Fluid velocity')
        eta.rename('Free surface displacement')

        u_.assign(u)
        eta_.assign(eta)

        if dumpn == ndump :

            dumpn -= ndump
            q_file.write(u, eta, time = t)
            # sol.interpolate(1e-3 * cos( - kap * t * sqrt(g * 0.1)) * exp( - (pow(x - 2., 2) + pow(y - 2., 2)) / 0.04))
            # ex_file.write(sol, time = t)        # TODO: ^^ Implement properly. Why doesn't it work??

            if mat_out == 'y' :
                M.rename('Metric field')
                m_file.write(M, time = t)
            else :
                print 't = %1.2fs' % t

# End timing and print:
toc1 = clock()
if remesh == 'y' :
    print 'Elapsed time for adaptive solver: %1.2fs' % (toc1 - tic1)
else :
    print 'Elapsed time for non-adaptive solver: %1.2fs' % (toc1 - tic1)