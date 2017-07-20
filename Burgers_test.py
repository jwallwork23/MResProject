from firedrake import *
from firedrake_adjoint import *

import numpy as np
from time import clock

from utils import construct_hessian, compute_steady_metric, interp

# Define initial (uniform) mesh:
n = int(raw_input('Mesh cells per m (default 16)?: ') or 16)            # Resolution of initial uniform mesh
lx = 4                                                                  # Extent in x-direction (m)
ly = 1                                                                  # Extent in y-direction (m)
mesh = RectangleMesh(lx * n, ly * n, lx, ly)
x, y = SpatialCoordinate(mesh)
N1 = len(mesh.coordinates.dat.data)                                     # Minimum number of nodes
N2 = N1                                                                 # Maximum number of nodes
print 'Initial number of nodes : ', N1

# Choose function space degree:
p = int(raw_input('Polynomial degree? (default 1): ') or 1)

# Specify diffusion:
nu = Constant(float(raw_input('Diffusion parameter (default 1e-3)?: ') or 1e-3))

# Set up adaptivity parameters:
remesh = raw_input('Use adaptive meshing (y/n)?: ') or 'y'
if remesh == 'y':
    hmin = float(raw_input('Minimum element size in mm (default 5)?: ') or 5.) * 1e-3
    hmax = float(raw_input('Maximum element size in mm (default 100)?: ') or 100.) * 1e-3
    rm = int(raw_input('Timesteps per remesh (default 5)?: ') or 5)
    nodes = float(raw_input('Target number of nodes (default 1000)?: ') or 1000.)
    ntype = raw_input('Normalisation type? (lp/manual): ') or 'lp'
    if ntype not in ('lp', 'manual'):
        raise ValueError('Please try again, choosing lp or manual.')
    mat_out = raw_input('Output Hessian and metric? (y/n): ') or 'n'
    if mat_out not in ('y', 'n'):
        raise ValueError('Please try again, choosing y or n.')
    hess_meth = raw_input('Integration by parts or double L2 projection? (parts/dL2): ') or 'dL2'
    if hess_meth not in ('parts', 'dL2'):
        raise ValueError('Please try again, choosing parts or dL2.')
else:
    hmin = 0.0625
    hmax = 0.3
    nodes = 0
    ntype = None
    hess_meth = None
    mat_out = 'n'

# Courant number adjusted timestepping parameters:
ndump = 1
T = 0.2
timestep = 0.8 * hmin
Dt = Constant(timestep)
print 'Using Courant number adjusted timestep dt = %1.4f' % timestep
if remesh == 'n':
    rm = int(T / timestep)

# Create function space and set initial conditions:
W = FunctionSpace(mesh, 'CG', p)
ic = Function(W)
ic.interpolate(1e-3 * exp(- (pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.04))
phi_ = ic.copy(deepcopy=True)
phi = Function(W, name='Concentration')
phi.assign(phi_)

# Initialise counters and files:
t = 0.
cnt = 0
dumpn = 0
mn = 0
phi_file = File('plots/adapt_plots/advection_test.pvd')
phi_file.write(phi, time=t)
m_file = File('plots/adapt_plots/advection_test_metric.pvd')
h_file = File('plots/adapt_plots/advection_test_hessian.pvd')
tic1 = clock()

# Enter timeloop:
while t < T - 0.5 * timestep:

    # Update counters:
    cnt += 1
    t += timestep
    dumpn += 1

    if (remesh == 'y') & (cnt % rm == 0):

        mn += 1

        # Compute Hessian and metric:
        V = TensorFunctionSpace(mesh, 'CG', 1)
        H = construct_hessian(mesh, V, phi, method=hess_meth)
        M = compute_steady_metric(mesh, V, H, phi, h_min=hmin, h_max=hmax, N=nodes, normalise=ntype)
        if mat_out == 'y':
            H.rename('Hessian')
            h_file.write(H, time=t)
            M.rename('Metric field')
            m_file.write(M, time=t)

        # Adapt mesh and interpolate functions:
        tic2 = clock()
        adaptor = AnisotropicAdaptation(mesh, M)
        mesh = adaptor.adapted_mesh
        phi_, phi = interp(adaptor, phi_, phi)
        W = FunctionSpace(mesh, 'CG', p)
        phi.rename('Concentration')
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
        print 'Elapsed time for adaption step %d: %1.2fs' % (mn, toc2 - tic2)
        print ''

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
    phi_solv.solve()
    phi_.assign(phi)

    if dumpn == ndump:
        dumpn -= ndump
        phi_file.write(phi, time=t)
        if remesh == 'n':
            print 't = %1.2fs' % t

    # End timing and print:
    toc1 = clock()
    if remesh == 'y':
        print 'Elapsed time for adaptive solver: %1.2fs' % (toc1 - tic1)
    else:
        print 'Elapsed time for non-adaptive solver: %1.2fs' % (toc1 - tic1)

# TODO: fix this
# Establish objective functional and compute derivatives:
J = Functional(inner(phi, phi) * dx * dt[FINISH_TIME])
print 'Functional established'
#dJdic = compute_gradient(J, Control(ic), forget=False) # Do we need to interpolate?
dJdnu = compute_gradient(J, Control(nu))
print 'Derivative computed!'
