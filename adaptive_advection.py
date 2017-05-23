from firedrake import *

import numpy as np
from time import clock

from utils import adapt, construct_hessian, compute_steady_metric, interp, update_advection_FE

# Define initial (uniform) mesh:
n = int(raw_input('Mesh cells per m (default 16)?: ') or 16)                    # Resolution of initial uniform mesh
lx = 4                                                                          # Extent in x-direction (m)
ly = 1                                                                          # Extent in y-direction (m)
mesh = RectangleMesh(lx * n, ly * n, lx, ly)
x, y = SpatialCoordinate(mesh)
print 'Initial number of nodes : ', len(mesh.coordinates.dat.data)

# Specify timestepping parameters:
ndump = int(raw_input('Timesteps per data dump (default 1): ') or 1)
T = 3.0                                                                         # Simulation end time (s)
dt = 0.1/(n * ndump)                                                            # Timestep length (s)
Dt = Constant(dt)

# Set up adaptivity parameters:
remesh = raw_input('Use adaptive meshing (y/n)?: ') or 'y'
if remesh == 'y' :
    hmin = float(raw_input('Minimum element size in mm (default 5)?: ') or 5.) * 1e-3
    rm = int(raw_input('Timesteps per remesh (default 5)?: ') or 5)
    nodes = float(raw_input('Target number of nodes (default 1000)?: ') or 1000.)
    ntype = raw_input('Normalisation type? (lp/manual): ') or 'lp'
else :
    hmin = 0
    rm = int(T / dt)
    nodes = 0
    ntype = None

# Create function space and set initial conditions:
W = FunctionSpace(mesh, 'CG', 1)
phi_ = Function(W)
phi_.interpolate(1e-3 * exp(-(pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.04))
phi = Function(W, name = 'Concentration')
phi.assign(phi_)
psi = TestFunction(W)

# Initialise counters and files:
t = 0.
mn = 0
dumpn = 0
phi_file = File('plots/adapt_plots/advection_test.pvd')
m_file = File('plots/adapt_plots/advection_test_metric.pvd')
phi_file.write(phi, time = t)
tic1 = clock()

# Enter timeloop:
while t < T - 0.5 * dt :

    # Update counters:
    mn += 1
    cnt = 0

    if remesh == 'y' :
        print '************ Adaption step %d **************' % mn

        # Compute Hessian and metric:
        V = TensorFunctionSpace(mesh, 'CG', 1)
        H = construct_hessian(mesh, V, phi)
        M = compute_steady_metric(mesh, V, H, phi, h_min = hmin, N = nodes)
        M.rename('Metric field')
        m_file.write(M, time = t)

        # Adapt mesh and set up new function spaces:
        mesh_ = mesh
        tic2 = clock()
        mesh = adapt(mesh, M)
        phi_, phi, W = update_advection_FE(mesh_, mesh, phi_, phi)
        toc2 = clock()
        print 'Number of nodes after adaption step %d: ' % mn, len(mesh.coordinates.dat.data)
        print 'Elapsed time for adaption step %d: %1.2es' % (mn, toc2 - tic2)
        phi.rename('Concentration')

    # Set up variational problem, using implicit midpoint timestepping:
    psi = TestFunction(W)
    phih = 0.5 * (phi + phi_)
    F = ((phi - phi_) * psi - Dt * phih * psi.dx(0)) * dx

    # Enter inner timeloop:
    while cnt < rm :
        t += dt
        cnt += 1
        solve(F == 0, phi, solver_parameters = {'pc_type' : 'ilu',
                                                'ksp_max_it' : 1500,})
        phi_.assign(phi)
        dumpn += 1
        if dumpn == ndump :
            dumpn -= ndump
            print 't = %1.2fs, mesh number = ' % t, mn
            phi_file.write(phi, time = t)

# End timing and print:
toc1 = clock()
if remesh == 'y' :
    print 'Elapsed time for adaptive solver: %1.2es' % (toc1 - tic1)
else :
    print 'Elapsed time for non-adaptive solver: %1.2es' % (toc1 - tic1)