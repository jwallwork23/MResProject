from firedrake import *

import numpy as np
from time import clock

from utils import adapt, construct_hessian, compute_steady_metric, interp

def update_setup(mesh1, mesh2, phi_, phi):
    """Update all functions from one mesh to another."""

    Vphi = FunctionSpace(mesh2, 'CG', 1)
    phi_2 = Function(Vphi)
    phi2 = Function(Vphi)
    interp(phi_, mesh1, phi_2, mesh2)
    interp(phi, mesh1, phi2, mesh2)

    return phi_2, phi2, Vphi

# Specify problem parameters:
T = 5.0
n = int(raw_input('Mesh cells per m (default 16)?: ') or 16)
dt = 0.1/n
Dt = Constant(dt)
remesh = raw_input('Use adaptive meshing (y/n)?: ') or 'y'
if remesh == 'y':
    hmin = float(raw_input('Minimum element size (default 0.005)?: ') or 0.005)
    rm = int(raw_input('Timesteps per remesh (default 5)?: ') or 5)
else:
    rm = int(T/dt)
    hmin = 0

# Define uniform mesh, with a metric function space:
lx = 4
ly = 1
mesh = PeriodicRectangleMesh(lx * n, ly * n, lx, ly)
print 'Initial number of nodes : ', len(mesh.coordinates.dat.data)

# Create functions and specify initial conditions:
W = FunctionSpace(mesh, 'CG', 1)
phi_ = Function(W)
phi_.interpolate(Expression('(x[0] > 0.5) & (x[0] < 1) ? -0.001 * sin(2 * pi * x[0]) : 0'))
phi = Function(W, name = 'Concentration')
phi.assign(phi_)
psi = TestFunction(W)

# Initialise counters and files:
t = 0.0
mn = 0
phi_file = File('plots/adapt_plots/advection_test.pvd')
m_file = File('plots/adapt_plots/advection_test_metric.pvd')
phi_file.write(phi, time = t)

# Enter timeloop:
while t < T - 0.5 * dt:

    # Update counters:
    mn += 1
    cnt = 0

    if remesh == 'y':

        # Compute Hessian and metric:
        V = TensorFunctionSpace(mesh, 'CG', 1)
        H = construct_hessian(mesh, V, phi)
        M = compute_steady_metric(mesh, V, H, phi, h_min = hmin)
        M.rename('Metric field')
        m_file.write(M, time = t)

        # Adapt mesh and set up new function spaces:
        mesh_ = mesh
        tic1 = clock()
        mesh = adapt(mesh_, M)
        toc1 = clock()
        print 'Elapsed time for adaption step {y}: %1.2es'.format(y=mn) % (toc1 - tic1)
        print 'Number of nodes after adaption step {y}: '.format(y=mn), len(mesh.coordinates.dat.data)
        phi_, phi, W = update_setup(mesh_, mesh, phi_, phi)
        phi.rename('Concentration')

    # Set up variational problem:
    psi = TestFunction(W)
    phih = 0.5 * (phi + phi_)
    F = ((phi - phi_) * psi - Dt * phih * psi.dx(0)) * dx

    # Enter inner timeloop:
    while cnt < rm:
        t += dt
        print 't = ', t, ' seconds, mesh number = ', mn
        cnt += 1
        solve(F == 0, phi, solver_parameters = {'pc_type' : 'ilu',
                                                'ksp_max_it' : 1500,})
        phi_.assign(phi)
        phi_file.write(phi)