from firedrake import *
import numpy as np
from time import clock

from utils.adaptivity import compute_steady_metric, construct_hessian, metric_intersection, metric_gradation
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
SumN = N1                                                       # Sum over vertex counts
print '...... mesh loaded. Initial number of vertices : ', N1

print ''
print 'Options...'
bathy = raw_input('Flat bathymetry or shelf break (f/s, default s?): ') or 's'
numVer = float(raw_input('Target vertex count as a proportion of the initial number? (default 0.1): ') or 0.1) * N1
hmin = float(raw_input('Minimum element size in mm (default 1)?: ') or 1.) * 1e-3
hmax = float(raw_input('Maximum element size in mm (default 1000)?: ') or 1000) * 1e-3
ntype = raw_input('Normalisation type? (lp/manual): ') or 'lp'
mat_out = bool(raw_input('Hit anything but enter to output Hessian and metric: ')) or False
iso = bool(raw_input('Hit anything but enter to use isotropic, rather than anisotropic: ')) or False
gradbdy = bool(raw_input('Hit anything but enter to gradate to initial boundaries: ')) or False
if gradbdy:
    beta = float(raw_input('Metric gradation scaling parameter (default 1.4): ') or 1.4)
if not iso:
    hess_meth = raw_input('Integration by parts or double L2 projection? (parts/dL2, default dL2): ') or 'dL2'

# Specify parameters:
depth = 0.1             # Water depth for flat bathymetry case (m)
ndump = 1               # Timesteps per data dump
T = 2.5                 # Simulation duration (s)
Ts = 0.5                # Time range lower limit (s), during which we can assume the wave won't reach the shore
g = 9.81                # Gravitational acceleration (m s^{-2})
dt = 0.05
Dt = Constant(dt)
rm = int(raw_input('Timesteps per re-mesh (default 5)?: ') or 5)
stored = bool(raw_input('Hit anything but enter if adjoint data is already stored: ')) or False

# Define mixed Taylor-Hood function space:
W = VectorFunctionSpace(mesh, 'CG', 2) * FunctionSpace(mesh, 'CG', 1)

# Interpolate bathymetry:
b = Function(W.sub(1), name='Bathymetry')
if bathy == 'f':
    b.interpolate(Expression(depth))
else:
    b.interpolate(Expression('x[0] <= 0.5 ? 0.01 : 0.1'))  # Shelf break bathymetry

# Check CFL criterion is satisfied for this discretisation:
assert(dt < 1. / (n * np.sqrt(g * max(b.dat.data))))

# Specify solver parameters:
params = {'mat_type': 'matfree',
          'snes_type': 'ksponly',
          'pc_type': 'python',
          'pc_python_type': 'firedrake.AssembledPC',
          'assembled_pc_type': 'lu',
          'snes_lag_preconditioner': -1,
          'snes_lag_preconditioner_persists': True}

# Initalise counters:
t = T
i = -1
dumpn = ndump
meshn = rm
tic2 = clock()

# Forcing switch:
coeff = Constant(1.)
switch = True

if not stored:
    # Create adjoint variables:
    lam_ = Function(W)
    lu_, le_ = lam_.split()

    # Establish indicator function for adjoint equations:
    f = Function(W.sub(1), name='Forcing term')
    f.interpolate(Expression('(x[0] >= 0.) & (x[0] < 0.25) & (x[1] > 1.8) & (x[1] < 2.2) ? 1e-3 : 0.'))

    # Interpolate adjoint final time conditions:
    lu_.interpolate(Expression([0, 0]))
    le_.assign(0)

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
    lam_file.write(lu, le, time=T)

    # Establish test functions and midpoint averages:
    w, xi = TestFunctions(W)
    lu, le = split(lam)
    lu_, le_ = split(lam_)
    luh = 0.5 * (lu + lu_)
    leh = 0.5 * (le + le_)

    # Set up the variational problem:
    La = ((le - le_) * xi - Dt * g * inner(luh, grad(xi)) - coeff * f * xi
          + inner(lu - lu_, w) + Dt * b * inner(grad(leh), w)) * dx
    lam_prob = NonlinearVariationalProblem(La, lam)
    lam_solv = NonlinearVariationalSolver(lam_prob, solver_parameters=params)

    # Split to access data:
    lu, le = lam.split()
    lu_, le_ = lam_.split()

    print ''
    print 'Starting fixed resolution adjoint run...'
while t > 0.5 * dt:

    # Increment counters:
    t -= dt
    dumpn -= 1
    meshn -= 1

    # Modify forcing term:
    if (t < Ts + 1.5 * dt) & switch:
        coeff.assign(0.5)
    if (t < Ts + 0.5 * dt) & switch:
        switch = False
        coeff.assign(0.)

    # Solve the problem and update:
    if not stored:
        lam_solv.solve()
        lam_.assign(lam)

        # Dump to vtu:
        if dumpn == 0:
            dumpn += ndump
            lam_file.write(lu, le, time=t)

        # Dump to HDF5:
        if meshn == 0:
            meshn += rm
            i -= 1
            # Interpolate velocity onto P1 space and store final time data to HDF5 and PVD:
            if not stored:
                print 't = %1.1fs' % t
                lu_P1.interpolate(lu)
                with DumbCheckpoint('data_dumps/tests/adjoint_soln_{y}'.format(y=i), mode=FILE_CREATE) as chk:
                    chk.store(lu_P1)
                    chk.store(le)
if not stored:
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
if iso:
    q_file = File('plots/goal-based_outputs/test_forward_iso.pvd')
    sig_file = File('plots/goal-based_outputs/test_significance_iso.pvd')
    if mat_out:
        m_file = File('plots/goal-based_outputs/SW_test_metric_iso.pvd')
        h_file = File('plots/goal-based_outputs/SW_test_hessian_iso.pvd')
else:
    q_file = File('plots/goal-based_outputs/test_forward.pvd')
    sig_file = File('plots/goal-based_outputs/test_significance.pvd')
    if mat_out:
        m_file = File('plots/goal-based_outputs/SW_test_metric.pvd')
        h_file = File('plots/goal-based_outputs/SW_test_hessian.pvd')
q_file.write(u, eta, time=0)

# Initialise counters:
t = 0.
dumpn = 0
mn = 0

# Approximate isotropic metric at boundaries of initial mesh using circumradius:
if gradbdy:
    h = Function(W.sub(1))
    h.interpolate(CellSize(mesh_))
    M_ = Function(TensorFunctionSpace(mesh_, 'CG', 1))
    for j in DirichletBC(W.sub(1), 0, 'on_boundary').nodes:
        h2 = pow(h.dat.data[j], 2)
        M_.dat.data[j][0, 0] = 1. / h2
        M_.dat.data[j][1, 1] = 1. / h2

print ''
print 'Starting mesh adaptive forward run...'
while t < T - 0.5 * dt:
    tic2 = clock()
    mn += 1

    # Interpolate velocity in a P1 space:
    vel = Function(VectorFunctionSpace(mesh, 'CG', 1))
    vel.interpolate(u)

    # Create functions to hold inner product and significance data:
    ip = Function(W.sub(1), name='Inner product')
    significance = Function(W.sub(1), name='Significant regions')

    # Take maximal L2 inner product as most significant:
    for j in range(max(i, int((Ts - T) / (dt * ndump))), 0):

        W = VectorFunctionSpace(mesh_, 'CG', 1) * FunctionSpace(mesh_, 'CG', 1)

        # Read in saved data from .h5:
        with DumbCheckpoint('data_dumps/tests/adjoint_soln_{y}'.format(y=i), mode=FILE_READ) as chk:
            lu_P1 = Function(W.sub(0), name='P1 adjoint velocity')
            le = Function(W.sub(1), name='Adjoint free surface')
            chk.load(lu_P1)
            chk.load(le)

        # Interpolate saved data onto new mesh:
        if mn != 1:
            print '    #### Interpolation step', j - max(i, int((Ts - T) / (dt * ndump))) + 1, '/',\
                len(range(max(i, int((Ts - T) / (dt * ndump))), 0))
            lu_P1, le = interp(mesh, lu_P1, le)

        # Multiply fields together:
        ip.dat.data[:] = lu_P1.dat.data[:, 0] * vel.dat.data[:, 0] + lu_P1.dat.data[:, 1] * vel.dat.data[:, 1] \
                         + le.dat.data * eta.dat.data

        # Extract (pointwise) maximal values:
        if j == 0:
            significance.dat.data[:] = ip.dat.data[:]
        else:
            for k in range(len(ip.dat.data)):
                if np.abs(ip.dat.data[k]) > np.abs(significance.dat.data[k]):
                    significance.dat.data[k] = ip.dat.data[k]
    sig_file.write(significance, time=t)

    # Generate Hessian associated with significant data:
    V = TensorFunctionSpace(mesh, 'CG', 1)
    H = Function(V)
    if iso:
        for i in range(len(H.dat.data)):
            H.dat.data[i][0, 0] = np.abs(significance.dat.data[i])
            H.dat.data[i][1, 1] = np.abs(significance.dat.data[i])
    else:
        H = construct_hessian(mesh, V, significance, method=hess_meth)
    M = compute_steady_metric(mesh, V, H, significance, h_min=hmin, h_max=hmax, normalise=ntype, num=numVer)

    if gradbdy:
        # Interpolate initial mesh size onto new mesh and build associated metric:
        fields = interp(mesh, h)
        W1 = FunctionSpace(mesh, 'CG', 1)
        h = Function(W1)
        h.dat.data[:] = fields[0].dat.data[:]
        M_ = Function(V)
        for j in DirichletBC(W1, 0, 'on_boundary').nodes:
            h2 = pow(h.dat.data[j], 2)
            M_.dat.data[j][0, 0] = 1. / h2
            M_.dat.data[j][1, 1] = 1. / h2

    # Gradate metric, adapt mesh and interpolate variables:
        M = metric_intersection(mesh, V, M, M_, bdy=True)
        metric_gradation(mesh, M, beta, isotropic=iso)
    adaptor = AnisotropicAdaptation(mesh, M)
    mesh = adaptor.adapted_mesh
    u, u_, eta, eta_, q, q_, b, W = interp_Taylor_Hood(mesh, u, u_, eta, eta_, b)
    u.rename('Fluid velocity')
    eta.rename('Free surface displacement')
    i += 1

    # Mesh resolution analysis:
    n = len(mesh.coordinates.dat.data)
    SumN += n
    if n < N1:
        N1 = n
    elif n > N2:
        N2 = n
    toc2 = clock()

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
            if mat_out:
                H.rename('Hessian')
                M.rename('Metric')
                h_file.write(H, time=t)
                m_file.write(M, time=t)

    # Print to screen:
    print ''
    print '************ Adaption step %d **************' % mn
    print 'Time = %1.1fs / %1.1fs' % (t, T)
    print 'Number of vertices after adaption step %d: ' % mn, n
    print 'Min/max vertex counts: %d, %d' % (N1, N2)
    print 'Mean vertex count: %d' % (float(SumN) / mn)
    print 'Elapsed time for this step: %1.2fs' % (toc2 - tic2)
    print ''
print '\a'
toc1 = clock()
print 'Elapsed time for adaptive solver: %1.1fs (%1.2f mins)' % (toc1 - tic1, (toc1 - tic1) / 60)
