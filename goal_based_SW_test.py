from firedrake import *
import numpy as np
from time import clock

from utils.adaptivity import compute_steady_metric, construct_hessian
from utils.interp import interp, interp_Taylor_Hood

print ''
print '******************************** SHALLOW WATER TEST PROBLEM ********************************'
print ''
print 'GOAL-BASED, mesh adaptive solver initially defined on a rectangular mesh'
tic1 = clock()

# Define initial (uniform) mesh:
n = 16
lx = 4                                                          # Extent in x-direction (m)
mesh = SquareMesh(lx * n, lx * n, lx, lx)
mesh_ = mesh
x, y = SpatialCoordinate(mesh)
N1 = len(mesh.coordinates.dat.data)                             # Minimum number of vertices
N2 = N1                                                         # Maximum number of vertices
print '...... mesh loaded. Initial number of nodes : ', N1
print ''
print 'Options...'
bathy = raw_input('Flat bathymetry or shelf break (f/s, default s?): ') or 's'

# Set up adaptivity parameters:
hmin = float(raw_input('Minimum element size in mm (default 5)?: ') or 5.) * 1e-3
hmax = float(raw_input('Maximum element size in mm (default 100)?: ') or 100) * 1e-3
ntype = raw_input('Normalisation type? (lp/manual): ') or 'lp'
if ntype not in ('lp', 'manual'):
    raise ValueError('Please try again, choosing lp or manual.')
mat_out = raw_input('Output Hessian and metric? (y/n, default n): ') or 'n'
if mat_out not in ('y', 'n'):
    raise ValueError('Please try again, choosing y or n.')
hess_meth = raw_input('Integration by parts or double L2 projection? (parts/dL2): ') or 'dL2'
if hess_meth not in ('parts', 'dL2'):
    raise ValueError('Please try again, choosing parts or dL2.')
nodes = N1              # Target number of vertices

# Specify parameters:
depth = 0.1             # Water depth for flat bathymetry case (m)
ndump = 1               # Timesteps per data dump
T = 2.5                 # Simulation duration (s)
Ts = 0.5                # Time range lower limit (s), during which we can assume the wave won't reach the shore
g = 9.81                # Gravitational acceleration (m s^{-2})
dt = 0.05
Dt = Constant(dt)
rm = int(raw_input('Timesteps per re-mesh (default 10)?: ') or 10)
stored = raw_input('Adjoint already computed? (y/n, default n): ') or 'n'

# Check CFL criterion is satisfied for this discretisation:
assert(dt < 1. / (n * np.sqrt(g * depth)))

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

# Interpolate bathymetry:
b = Function(W.sub(1), name='Bathymetry')
if bathy == 'f':
    b.interpolate(Expression(depth))
else:
    b.interpolate(Expression('x[0] <= 0.5 ? 0.01 : 0.1'))  # Shelf break bathymetry

# Initalise counters:
t = T
i = -1
dumpn = ndump
meshn = rm

if stored == 'n':
    # Create adjoint variables:
    lam_ = Function(W)
    lu_, le_ = lam_.split()

    # Establish indicator function for adjoint equations:
    f = Function(W.sub(1), name='Forcing term')
    f.interpolate(Expression('(x[0] >= 0.) & (x[0] < 0.25) & (x[1] > 1.8) & (x[1] < 2.2) ? 1e-3 : 0.'))

    # Interpolate adjoint final time conditions:
    lu_.interpolate(Expression([0, 0]))
    le_.assign(f)

    # Set up dependent variables of the adjoint problem:
    lam = Function(W)
    lam.assign(lam_)
    lu, le = lam.split()
    lu.rename('Adjoint velocity')
    le.rename('Adjoint free surface')

    # Interpolate velocity onto P1 space and store final time data to HDF5 and PVD:
    lu_P1 = Function(VectorFunctionSpace(mesh, 'CG', 1), name='P1 adjoint velocity')
    lu_P1.interpolate(lu)
    with DumbCheckpoint('data_dumps/tests/adjoint_soln_{y}'.format(y=i), mode=FILE_CREATE) as chk:
        chk.store(lu_P1)
        chk.store(le)
    lam_file = File('plots/goal-based_outputs/test_adjoint.pvd')
    lam_file.write(lu, le, time=0)

    # Establish test functions and midpoint averages:
    w, xi = TestFunctions(W)
    lu, le = split(lam)
    lu_, le_ = split(lam_)
    luh = 0.5 * (lu + lu_)
    leh = 0.5 * (le + le_)

    # Set up the variational problem:
    La = ((le - le_) * xi - Dt * g * inner(luh, grad(xi)) - f * xi
          + inner(lu - lu_, w) + Dt * b * inner(grad(leh), w)) * dx
    lam_prob = NonlinearVariationalProblem(La, lam)
    lam_solv = NonlinearVariationalSolver(lam_prob, solver_parameters=params)

    # Split to access data:
    lu, le = lam.split()
    lu_, le_ = lam_.split()

    print ''
    print 'Starting fixed resolution adjoint run...'
    tic2 = clock()
while t > 0.5 * dt:
    print 'i = ', i

    # Increment counters:
    t -= dt
    dumpn -= 1
    meshn -= 1

    # Solve the problem and update:
    if stored == 'n':
        lam_solv.solve()
        lam_.assign(lam)

    # Dump to vtu:
    if dumpn == 0:
        dumpn += ndump
        lam_file.write(lu, le, time=T - t)

    # Dump to HDF5:
    if meshn == 0:
        meshn += rm
        i -= 1
        # Interpolate velocity onto P1 space and store final time data to HDF5 and PVD:
        if stored == 'n':
            print 't = %1.1fs' % t
            lu_P1.interpolate(lu)
            with DumbCheckpoint('data_dumps/tests/adjoint_soln_{y}'.format(y=i), mode=FILE_CREATE) as chk:
                chk.store(lu_P1)
                chk.store(le)
if stored == 'n':
    print '... done!',
    toc2 = clock()
    print 'Elapsed time for adjoint solver: %1.2fs' % (toc2 - tic2)

# Repeat above setup:
q_ = Function(W)
u_, eta_ = q_.split()

# Interpolate forward initial conditions:
u_.interpolate(Expression([0, 0]))
eta_.interpolate(1e-3 * exp(- (pow(x - 2., 2) + pow(y - 2., 2)) / 0.04))

# Set up dependent variables of the forward problem:
q = Function(W)
q.assign(q_)
u, eta = q.split()
u.rename('Fluid velocity')
eta.rename('Free surface displacement')

# Intialise files:
q_file = File('plots/goal-based_outputs/test_forward.pvd')
q_file.write(u, eta, time=0)
sig_file = File('plots/goal-based_outputs/test_significance.pvd')
m_file = File('plots/goal-based_outputs/SW_test_metric.pvd')
h_file = File('plots/goal-based_outputs/SW_test_hessian.pvd')

# Initialise counters:
t = 0.
dumpn = 0
i0 = i

print ''
print 'Starting mesh adaptive forward run...'
while t < T - 0.5 * dt:
    tic2 = clock()
    print 'i = ', i
    # Interpolate velocity in a P1 space:
    vel = Function(VectorFunctionSpace(mesh, 'CG', 1))
    vel.interpolate(u)

    # Create functions to hold inner product and significance data:
    ip = Function(W.sub(1), name='Inner product')
    significance = Function(W.sub(1), name='Significant regions')

    # Take maximal L2 inner product as most significant:
    for j in range(max(i, int((Ts - T) / (dt * ndump))), 0):

        W = VectorFunctionSpace(mesh_, 'CG', 1) * FunctionSpace(mesh_, 'CG', 1)

        with DumbCheckpoint('data_dumps/tests/adjoint_soln_{y}'.format(y=i), mode=FILE_READ) as chk:
            lu_P1 = Function(W.sub(0), name='P1 adjoint velocity')
            le = Function(W.sub(1), name='Adjoint free surface')
            chk.load(lu_P1)
            chk.load(le)

        # Interpolate saved data onto new mesh:
        if (i + int(T / (dt * ndump))) != 0:
            print '    #### Interpolation step', j - max(i, int((Ts - T) / (dt * ndump))) + 1, '/',\
                len(range(max(i, int((Ts - T) / (dt * ndump))), 0))
            lu_P1, le = interp(mesh, lu_P1, le)

        ip.dat.data[:] = lu_P1.dat.data[:, 0] * vel.dat.data[:, 0] + lu_P1.dat.data[:, 1] * vel.dat.data[:, 1] \
                         + le.dat.data * eta.dat.data
        if (j == 0) | (np.abs(assemble(ip * dx)) > np.abs(assemble(significance * dx))):
            significance.dat.data[:] = ip.dat.data[:]
    sig_file.write(significance, time=t)

    # Adapt mesh to significant data and interpolate:
    V = TensorFunctionSpace(mesh, 'CG', 1)
    H = construct_hessian(mesh, V, significance)
    M = compute_steady_metric(mesh, V, H, significance, h_min=0.1, h_max=5)
    adaptor = AnisotropicAdaptation(mesh, M)
    mesh = adaptor.adapted_mesh
    u, u_, eta, eta_, q, q_, b, W = interp_Taylor_Hood(mesh, u, u_, eta, eta_, b)
    u.rename('Fluid velocity')
    eta.rename('Free surface displacement')
    i += 1

    # Mesh resolution analysis:
    n = len(mesh.coordinates.dat.data)
    if n < N1:
        N1 = n
    elif n > N2:
        N2 = n
    toc2 = clock()

    # Print to screen:
    print ''
    print '************ Adaption step %d **************' % (i + i0)
    print 'Time = %1.2fs / %1.1fs' % (t, T)
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
            H.rename('Hessian')
            M.rename('Metric')
            h_file.write(H, time=t)
            m_file.write(M, time=t)

toc1 = clock()
print 'Elapsed time for adaptive solver: %1.1f minutes' % ((toc1 - tic1) / 60.)
