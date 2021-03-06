from firedrake import *
import numpy as np
from time import clock

import utils.adaptivity as adap
import utils.interp as inte

print('\n******************************** SHALLOW WATER TEST PROBLEM ********************************\n')
print('Mesh adaptive solver initially defined on a square mesh')
tic1 = clock()

# Define initial (uniform) mesh:
n = 16                                                          # Resolution of initial uniform mesh
lx = 4                                                          # Extent in x-direction (m)
mesh = SquareMesh(lx * n, lx * n, lx, lx)
x, y = SpatialCoordinate(mesh)
N1 = len(mesh.coordinates.dat.data)                             # Minimum number of vertices
N2 = N1                                                         # Maximum number of vertices
SumN = N1                                                       # Sum over vertex counts
print('...... mesh loaded. Initial number of vertices : ', N1, '\nOptions...')

bathy = input('Flat bathymetry or shelf break (f/s, default s)?: ') or 's'
numVer = float(input('Target vertex count as a proportion of the initial number? (default 0.2): ') or 0.2) * N1
hmin = float(input('Minimum element size in mm (default 1)?: ') or 1.) * 1e-3
hmax = float(input('Maximum element size in mm (default 1000)?: ') or 1000.) * 1e-3
hmin2 = pow(hmin, 2)      # Square minimal side-length
hmax2 = pow(hmax, 2)      # Square maximal side-length
ntype = input('Normalisation type? (lp/manual, default lp): ') or 'lp'
mtype = input('Adapt with respect to speed, free surface or both? (s/f/b, default b): ') or 'b'
if mtype not in ('s', 'f', 'b'):
    raise ValueError('Field selection not recognised. Please try again, choosing s, f or b.')
mat_out = bool(input('Hit anything but enter to output Hessian and metric: ')) or False
iso = bool(input('Hit anything but enter to use isotropic, rather than anisotropic: ')) or False
if not iso:
    hess_meth = input('Integration by parts or double L2 projection? (parts/dL2, default dL2): ') or 'dL2'

# Courant number adjusted timestepping parameters:
depth = 0.1             # Water depth for flat bathymetry case (m)
ndump = 1               # Timesteps per data dump
T = 2.5                 # Simulation duration (s)
g = 9.81                # Gravitational acceleration (m s^{-2})
dt = 0.05
Dt = Constant(dt)
rm = int(input('Timesteps per re-mesh (default 5)?: ') or 5)

# Check CFL criterion is satisfied for this discretisation:
assert(dt < 1. / (n * np.sqrt(g * depth)))

# Define mixed Taylor-Hood function space and interpolate initial conditions:
W = VectorFunctionSpace(mesh, 'CG', 2) * FunctionSpace(mesh, 'CG', 1)
q_ = Function(W)
u_, eta_ = q_.split()
u_.interpolate(Expression([0, 0]))
eta_.interpolate(1e-3 * exp(- (pow(x - 2., 2) + pow(y - 2., 2)) / 0.04))

# Establish bathymetry function:
b = Function(W.sub(1), name='Bathymetry')
if bathy == 'f':
    b.assign(depth)                                             # Constant depth
else:
    b.interpolate(Expression('x[0] <= 0.5 ? 0.01 : 0.1'))       # Shelf break bathymetry

# Set up dependent variables of problem:
q = Function(W)
q.assign(q_)
u, eta = q.split()
u.rename('Fluid velocity')
eta.rename('Free surface displacement')

# Initialise files:
if iso:
    q_file = File('plots/isotropic_outputs/SW_test.pvd')
    if mat_out:
        m_file = File('plots/isotropic_outputs/SW_test_metric.pvd')
        h_file = File('plots/isotropic_outputs/SW_test_hessian.pvd')
else:
    q_file = File('plots/anisotropic_outputs/SW_test.pvd')
    if mat_out:
        m_file = File('plots/anisotropic_outputs/SW_test_metric.pvd')
        h_file = File('plots/anisotropic_outputs/SW_test_hessian.pvd')
q_file.write(u, eta, time=0)

# Initialise counters:
t = 0.
dumpn = 0
mn = 0

print('\nEntering outer timeloop!')
while t < T - 0.5 * dt:
    mn += 1
    tic2 = clock()

    # Compute Hessian and metric:
    V = TensorFunctionSpace(mesh, 'CG', 1)
    if iso:
        M = Function(V)
        if mtype == 's':
            spd2 = Function(FunctionSpace(mesh, 'CG', 1))
            spd2.interpolate(dot(u, u))
            for i in range(len(M.dat.data)):
                ispd2 = 1. / max(hmin2, min(spd2.dat.data[i], hmax2))
                M.dat.data[i][0, 0] = ispd2
                M.dat.data[i][1, 1] = ispd2
        elif mtype == 'f':
            for i in range(len(M.dat.data)):
                ieta2 = 1. / max(hmin2, min(pow(eta.dat.data[i], 2), hmax2))
                M.dat.data[i][0, 0] = ieta2
                M.dat.data[i][1, 1] = ieta2
        else:
            raise NotImplementedError('Cannot currently interpret isotropic adaption with respect to two fields.')
    else:
        H = Function(V)
        if mtype != 'f':
            spd = Function(FunctionSpace(mesh, 'CG', 1))
            spd.interpolate(sqrt(dot(u, u)))
            H = adap.construct_hessian(mesh, V, spd, method=hess_meth)
            M = adap.compute_steady_metric(mesh, V, H, spd, h_min=hmin, h_max=hmax, num=numVer, normalise=ntype)
        if mtype != 's':
            H = adap.construct_hessian(mesh, V, eta, method=hess_meth)
            M2 = adap.compute_steady_metric(mesh, V, H, eta, h_min=hmin, h_max=hmax, num=numVer, normalise=ntype)
        if mtype == 'b':
            M = adap.metric_intersection(mesh, V, M, M2)
        else:
            M = Function(V)
            M.assign(M2)

    # Adapt mesh with respect to computed metric field and interpolate functions onto new mesh:
    adaptor = AnisotropicAdaptation(mesh, M)
    mesh = adaptor.adapted_mesh
    u, u_, eta, eta_, q, q_, b, W = inte.interp_Taylor_Hood(mesh, u, u_, eta, eta_, b)

    # Mesh resolution analysis:
    n = len(mesh.coordinates.dat.data)
    SumN += 1
    if n < N1:
        N1 = n
    elif n > N2:
        N2 = n

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
    # Split to access data and relabel functions:
    u, eta = q.split()
    u_, eta_ = q_.split()
    u.rename('Fluid velocity')
    eta.rename('Free surface displacement')

    # Inner timeloop:
    for j in range(rm):
        t += dt
        dumpn += 1
        q_solv.solve()  # Solve problem
        q_.assign(q)    # Update variables

        if dumpn == ndump:
            dumpn -= ndump
            q_file.write(u, eta, time=t)
            if mat_out:
                H.rename('Hessian')
                M.rename('Metric')
                h_file.write(H, time=t)
                m_file.write(M, time=t)
    toc2 = clock()

    # Print to screen:
    print('\n************ Adaption step %d **************' % mn)
    print('Time = %1.1fs / %1.1fs' % (t, T))
    print('Number of vertices after adaption step %d: ' % mn, n)
    print('Min/max vertex counts: %d, %d' % (N1, N2))
    print('Mean vertex count: %d' % (float(SumN) / mn))
    print('Elapsed time for this step: %1.2fs' % (toc2 - tic2), '\n')
print('\a')
toc1 = clock()
print('Elapsed time for adaptive solver: %1.1fs (%1.2f mins)' % (toc1 - tic1, (toc1 - tic1) / 60))
