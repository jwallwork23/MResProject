from firedrake import *
import numpy as np
from time import clock

from utils.adaptivity import construct_hessian, compute_steady_metric
from utils.interp import interp

print('\n******************************** BURGERS EQUATION TEST PROBLEM ********************************\n')
print('Mesh adaptive solver initially defined on a rectangular mesh')
tic1 = clock()

# Define initial (uniform) mesh:
n = 16                                                                  # Resolution of initial uniform mesh
lx = 4                                                                  # Extent in x-direction (m)
ly = 1                                                                  # Extent in y-direction (m)
mesh = RectangleMesh(lx * n, ly * n, lx, ly)
x, y = SpatialCoordinate(mesh)
N1 = len(mesh.coordinates.dat.data)                                     # Minimum number of nodes
N2 = N1                                                                 # Maximum number of nodes
SumN = N1                                                               # Sum over vertex counts
print('...... mesh loaded. Initial number of nodes : ', N1)

# Choose function space degree:
print('\nOptions...')
numVer = float(raw_input('Target vertex count as a proportion of the initial number? (default 0.85): ') or 0.85) * N1
p = int(raw_input('Polynomial degree? (default 1): ') or 1)
nu = Constant(float(raw_input('Diffusion parameter (default 1e-3)?: ') or 1e-3))
hmin = float(raw_input('Minimum element size in mm (default 5)?: ') or 5.) * 1e-3
hmax = float(raw_input('Maximum element size in mm (default 100)?: ') or 100.) * 1e-3
ntype = raw_input('Normalisation type? (lp/manual): ') or 'lp'
if ntype not in ('lp', 'manual'):
    raise ValueError('Please try again, choosing lp or manual.')
mat_out = raw_input('Output Hessian and metric? (y/n): ') or 'n'
iso = bool(raw_input('Hit anything but enter to use isotropic, rather than anisotropic: ')) or False
if not iso:
    hess_meth = raw_input('Integration by parts or double L2 projection? (parts/dL2, default dL2): ') or 'dL2'
    if hess_meth not in ('parts', 'dL2'):
        raise ValueError('Please try again, choosing parts or dL2.')

# Courant number adjusted timestepping parameters:
ndump = 1
T = 0.2
dt = 0.8 * hmin
Dt = Constant(dt)
print('Using Courant number adjusted timestep dt = %1.4f' % dt)
rm = int(raw_input('Timesteps per remesh (default 5)?: ') or 5)

# Create function space and set initial conditions:
W = FunctionSpace(mesh, 'CG', p)
phi_ = Function(W)
phi_.interpolate(1e-3 * exp(- (pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.04))
phi = Function(W, name='Concentration')
phi.assign(phi_)

# Initialise counters and files:
t = 0.
dumpn = 0
mn = 0
if iso:
    phi_file = File('plots/isotropic_outputs/Burgers_test.pvd')
    m_file = File('plots/isotropic_outputs/Burgers_test_metric.pvd')
    h_file = File('plots/isotropic_outputs/Burgers_test_hessian.pvd')
else:
    phi_file = File('plots/anisotropic_outputs/Burgers_test.pvd')
    m_file = File('plots/anisotropic_outputs/Burgers_test_metric.pvd')
    h_file = File('plots/anisotropic_outputs/Burgers_test_hessian.pvd')
phi_file.write(phi, time=t)

print('\nEntering outer timeloop!')
while t < T - 0.5 * dt:
    mn += 1

    # Compute Hessian and metric:
    tic2 = clock()
    V = TensorFunctionSpace(mesh, 'CG', 1)
    H = Function(V)
    if iso:
        for i in range(len(H.dat.data)):
            H.dat.data[i][0, 0] = np.abs(phi.dat.data[i])
            H.dat.data[i][1, 1] = np.abs(phi.dat.data[i])
    else:
        H = construct_hessian(mesh, V, phi, method=hess_meth)
    M = compute_steady_metric(mesh, V, H, phi, h_min=hmin, h_max=hmax, num=nodes, normalise=ntype)
    if mat_out == 'y':
        H.rename('Hessian')
        h_file.write(H, time=t)
        M.rename('Metric field')
        m_file.write(M, time=t)

    # Adapt mesh and interpolate functions:
    adaptor = AnisotropicAdaptation(mesh, M)
    mesh = adaptor.adapted_mesh
    phi_, phi = interp(mesh, phi_, phi)
    W = FunctionSpace(mesh, 'CG', p)
    phi.rename('Concentration')
    toc2 = clock()

    # Mesh resolution analysis:
    n = len(mesh.coordinates.dat.data)
    SumN += n
    if n < N1:
        N1 = n
    elif n > N2:
        N2 = n

    # Set up variational problem, using implicit midpoint timestepping, and solve:
    psi = TestFunction(W)
    phih = 0.5 * (phi + phi_)
    F = ((phi - phi_) * psi - Dt * phih * psi.dx(0) + Dt * nu * inner(grad(phih), grad(psi))) * dx
    bc = DirichletBC(W, 0.0, 'on_boundary')
    phi_prob = NonlinearVariationalProblem(F, phi, bcs=bc)
    phi_solv = NonlinearVariationalSolver(phi_prob, solver_parameters={'mat_type': 'matfree',
                                                                       'snes_type': 'ksponly',
                                                                       'pc_type': 'python',
                                                                       'pc_python_type': 'firedrake.AssembledPC',
                                                                       'assembled_pc_type': 'lu',
                                                                       'snes_lag_preconditioner': -1,
                                                                       'snes_lag_preconditioner_persists': True})
    # Inner timeloop:
    for j in range(rm):
        t += dt
        dumpn += 1

        # Sole the problem and update:
        phi_solv.solve()
        phi_.assign(phi)

        if dumpn == ndump:
            dumpn -= ndump
            phi_file.write(phi, time=t)

    # Print to screen:
    print('\n************ Adaption step %d **************' % mn)
    print('Time = %1.1fs / %1.1fs' % (t, T))
    print('Number of vertices after adaption step %d: ' % mn, n)
    print('Min/max vertex counts: %d, %d' % (N1, N2))
    print('Mean vertex count: %d' % (float(SumN) / mn))
    print('Elapsed time for this step: %1.2fs' % (toc2 - tic2), '\n')
print('\a')
toc1 = clock()
print('Elapsed time for adaptive solver: %1.2fs' % (toc1 - tic1))
