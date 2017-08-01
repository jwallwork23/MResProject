from firedrake import *
import numpy as np
from time import clock

from utils.adaptivity import compute_steady_metric, construct_hessian
from utils.interp import interp, interp_Taylor_Hood

print ''
print '******************************** SHALLOW WATER TEST PROBLEM ********************************'
print ''
tic1 = clock()

# Define initial (uniform) mesh:
lx = 4e5                                                                        # Extent in x-direction (m)
mesh = SquareMesh(64, 64, lx, lx)
mesh_ = mesh
x, y = SpatialCoordinate(mesh)
N1 = len(mesh.coordinates.dat.data)                                             # Minimum number of vertices
N2 = N1                                                                         # Maximum number of vertices
print '...... mesh loaded. Initial number of nodes : ', N1
bathy = raw_input('Flat bathymetry or shelf break (f/s, default f)?: ') or 'f'

# Simulation duration:
T = 1e3

# Set up adaptivity parameters:
hmin = float(raw_input('Minimum element size in m (default 1)?: ') or 1.)
hmax = float(raw_input('Maximum element size in m (default 1000)?: ') or 1e3)
ntype = raw_input('Normalisation type? (lp/manual): ') or 'lp'
if ntype not in ('lp', 'manual'):
    raise ValueError('Please try again, choosing lp or manual.')
mtype = raw_input('Mesh w.r.t. speed, free surface or both? (s/f/b, defualt f): ') or 'f'
if mtype not in ('s', 'f', 'b'):
    raise ValueError('Please try again, choosing s, f or b.')
mat_out = raw_input('Output Hessian and metric? (y/n, default n): ') or 'n'
if mat_out not in ('y', 'n'):
    raise ValueError('Please try again, choosing y or n.')
hess_meth = raw_input('Integration by parts or double L2 projection? (parts/dL2): ') or 'dL2'
if hess_meth not in ('parts', 'dL2'):
    raise ValueError('Please try again, choosing parts or dL2.')
nodes = 0.5 * N1                # Target number of vertices

# Courant number adjusted timestepping parameters:
ndump = 50
g = 9.81                                                # Gravitational acceleration (m s^{-2})
dt = hmin / np.sqrt(g * 0.1)                            # Timestep length (s), using wavespeed sqrt(gh)
Dt = Constant(dt)
print 'Using Courant number adjusted timestep dt = %1.4f' % dt
rm = int(raw_input('Timesteps per re-mesh (default 100)?: ') or 100)
Ts = 250.                 # Time range lower limit (s), during which we can assume the wave won't reach the shore

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
    b.interpolate(Expression(4000.))
else:
    b.interpolate(Expression('x[0] <= 50e3 ? 200. : 4000.'))      # Shelf break bathymetry

# Create adjoint variables:
lam_ = Function(W)
lu_, le_ = lam_.split()

# Establish indicator function for adjoint equations:
f = Function(W.sub(1), name='Forcing term')
f.interpolate(Expression('(x[0] >= 10e3) & (x[0] < 25e3) & (x[1] > 180e3) & (x[1] < 220e3) ? 1. : 0.'))

# Interpolate adjoint final time conditions:
lu_.interpolate(Expression([0, 0]))
le_.assign(f)

# Set up dependent variables of the adjoint problem:
lam = Function(W)
lam.assign(lam_)
lu, le = lam.split()
vel = Function(VectorFunctionSpace(mesh, 'CG', 1))          # For interpolating velocity field

# Label variables:
lu.rename('Adjoint velocity')
le.rename('Adjoint free surface')

# Initialise files:
lam_file = File('plots/goal-based_outputs/test_adjoint.pvd')
lam_file.write(lu, le, time=0)

# Initalise counters:
t = T
i = 0
dumpn = ndump

# Initialise tensor arrays for storage (with dimensions pre-allocated for speed):
sol_dat = np.zeros((int(T / (dt * ndump)) + 1, N1, 3))
significant_dat = np.zeros((int(T / (dt * ndump)) + 1, N1))
vel.interpolate(lu)
sol_dat[i, :, :2] = vel.dat.data
sol_dat[i, :, 2] = le.dat.data

# Establish test functions and midpoint averages:
w, xi = TestFunctions(W)
lu, le = split(lam)
lu_, le_ = split(lam_)
luh = 0.5 * (lu + lu_)
leh = 0.5 * (le + le_)

# Set up the variational problem:
La = ((le - le_) * xi - Dt * g * b * inner(luh, grad(xi)) - f * xi
      + inner(lu - lu_, w) + Dt * b * inner(grad(leh), w)) * dx
lam_prob = NonlinearVariationalProblem(La, lam)
lam_solv = NonlinearVariationalSolver(lam_prob, solver_parameters=params)

# Split to access data and relabel functions:
lu, le = lam.split()
lu_, le_ = lam_.split()

print ''
print 'Starting fixed resolution adjoint run...'
tic2 = clock()
while t > 0.5 * dt:

    # Increment counters:
    t -= dt
    dumpn -= 1

    # Solve the problem and update:
    lam_solv.solve()
    lam_.assign(lam)

    # Dump to vtu:
    if dumpn == 0:
        i -= 1
        dumpn += ndump

        # Save data:
        vel.interpolate(lu)
        sol_dat[i, :, :2] = vel.dat.data
        sol_dat[i, :, 2] = le.dat.data

        lam_file.write(lu, le, time=T-t)
        print 't = %1.1fs' % t
print '... done!',
toc2 = clock()
print 'Elapsed time for adjoint solver: %1.2fs' % (toc2 - tic2)

# Repeat above setup:
q_ = Function(W)
u_, eta_ = q_.split()

# Interpolate forward initial conditions:
u_.interpolate(Expression([0, 0]))
eta_.interpolate(exp(- (pow(x - 200e3, 2) + pow(y - 200e3, 2)) / 200.))

# Set up dependent variables of the forward problem:
q = Function(W)
q.assign(q_)
u, eta = q.split()

# Label variables:
u.rename('Fluid velocity')
eta.rename('Free surface displacement')

# Intialise files:
q_file = File('plots/goal-based_outputs/test_forward.pvd')
q_file.write(u, eta, time=0)
sig_file = File('plots/goal-based_outputs/test_significance.pvd')

# Initialise counters:
t = 0.
dumpn = 0

print ''
print 'Starting mesh adaptive forward run...'
while t < T - 0.5 * dt:
    tic2 = clock()

    # Interpolate velocity in a P1 space:
    vel.interpolate(u)

    # Create functions to hold inner product and significance data:
    ip = Function(W.sub(1), name='Inner product')
    significance = Function(W.sub(1), name='Significant regions')

    # Take maximal L2 inner product as most significant:
    for j in range(max(i, int((Ts - T) / (dt * ndump))), 0):

        sol_v = Function(VectorFunctionSpace(mesh_, 'CG', 1))
        sol_e = Function(FunctionSpace(mesh_, 'CG', 1))
        sol_v.dat.data[:, :] = sol_dat[j, :, :2]
        sol_e.dat.data[:] = sol_dat[j, :, 2]

        # Interpolate saved data onto new mesh:
        if (i + int(T / (dt * ndump))) != 0:
            print '#### Interpolation step', j - max(i, int((Ts - T) / (dt * ndump))) + 1, '/',\
                len(range(max(i, int((Ts - T) / (dt * ndump))), 0))
            sol_v, sol_e = interp(mesh, sol_v, sol_e)

        ip.dat.data[:] = sol_v.dat.data[:, 0] * vel.dat.data[:, 0] + sol_v.dat.data[:, 1] * vel.dat.data[:, 1] \
                         + sol_e.dat.data * eta.dat.data
        if (j == 0) | (np.abs(assemble(ip * dx)) > np.abs(assemble(significance * dx))):
            significance.dat.data[:] = ip.dat.data[:]

    # Adapt mesh to significant data and interpolate:
    V = TensorFunctionSpace(mesh, 'CG', 1)
    H = construct_hessian(mesh, V, significance)
    M = compute_steady_metric(mesh, V, H, significance, h_min=0.1, h_max=5)
    adaptor = AnisotropicAdaptation(mesh, M)
    mesh = adaptor.adapted_mesh
    u, u_, eta, eta_, q, q_, b, W = interp_Taylor_Hood(mesh, u, u_, eta, eta_, b)
    vel = Function(VectorFunctionSpace(mesh, 'CG', 1))
    u.rename('Fluid velocity')
    eta.rename('Free surface displacement')

    # Data analysis:
    n = len(mesh.coordinates.dat.data)
    if n < N1:
        N1 = n
    elif n > N2:
        N2 = n
    toc2 = clock()

    # Print to screen:
    print ''
    print '************ Adaption step %d **************' % (i + 40)
    print 'Time = %1.2fs, i = %d' % (t, i)
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
            i += 1
            q_file.write(u, eta, time=t)
            sig_file.write(significance, time=t)

toc1 = clock()
print 'Elapsed time for adaptive solver: %1.2fs' % (toc1 - tic1)
