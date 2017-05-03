from firedrake import *
import numpy as np
from numpy import linalg as LA

from adaptivity import *
from domain import tank_domain
from forms import *
from interp import interp

# Specify problem parameters:
mode = raw_input('Use linear or nonlinear equations? (l/n): ') or 'l'
if ((mode != 'l') & (mode != 'n')):
    raise ValueError('Please try again, choosing l or n.')
dt = float(raw_input('Timestep (default 0.1)?: ') or 0.1)
Dt = Constant(dt)
n = int(raw_input('Number of mesh cells per m (default 16)?: ') or 16)
n2 = n**2
T = float(raw_input('Simulation duration in s (default 5)?: ') or 5.0)
remesh = raw_input('Use adaptive meshing (y/n)?: ') or 'y'
if ((remesh != 'y') & (remesh != 'n')):
    raise ValueError('Please try again, typing y or n.')

# Set numerical parameters for the scheme:
ndump = 1       # Timesteps per data dump
rm = 6          # Timesteps per remesh

# Establish tank domain
mesh, Vq, q_, u_, eta_, lam_, lu_, le_, b, BCs = tank_domain(n)

# Initialisation:
t = 0.0; dumpn = 0; mn = 0
q = Function(Vq)
q.assign(q_)
q_file = File('plots/prob1_test_outputs/prob1_adapt.pvd'.format(y=mn))
params = {'mat_type': 'matfree',
          'snes_type': 'ksponly',
          'pc_type': 'python',
          'pc_python_type': 'firedrake.AssembledPC',
          'assembled_pc_type': 'lu',
          'snes_lag_preconditioner': -1,
          'snes_lag_preconditioner_persists': True,}

while (t < T-0.5*dt):

    mn += 1

    if (t != 0.0):

        # Set up metric:
        Vm = TensorFunctionSpace(mesh, 'CG', 1)
        M = Function(Vm)

        # Build Hessian and (hence) metric:
        H, V = construct_hessian(mesh, eta)
        if (remesh == 'y'):
            M = compute_steady_metric(mesh, V, H, eta)
        else:
            M.interpolate(Expression([[n2, 0], [0, n2]]))

        # Adapt mesh and update FE setup:
        mesh_ = mesh
        mesh = adapt(mesh, M)
        q_, q, u_, u, eta_, eta, b, Vq = update_SW_FE(mesh_, mesh, u_, u, eta_, eta, b)

    # Establish variational problem:
    if (mode == 'l'):
        L = linear_form
    else:
        L = nonlinear_form
    q_, q, u_, u, eta_, eta, q_solv = SW_solve(q_, q, u_, eta_, b, Dt, Vq, params, L)

    if (t == 0.0):
        q_file.write(u, eta, time=t)
    cnt = 0

    # Enter the inner timeloop:
    while (cnt < rm):     
        t += dt
        print 't = ', t, ' seconds, mesh number = ', mn
        cnt += 1
        q_solv.solve()
        q_.assign(q)
        dumpn += 1
        if (dumpn == ndump):
            dumpn -= ndump
            q_file.write(u, eta, time=t)
