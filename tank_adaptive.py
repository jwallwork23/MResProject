from firedrake import *
import pyop2 as op2             # Not currently used

import sys, os, os.path         # Not currently used
import numpy as np
from numpy import linalg as LA

from adaptivity import *
from domain import *
from forms import *
from interp import *

# Specify problem parameters:
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
mesh, Vq, q_, u_, eta_, b = tank_domain_wp(n)   # TODO: use Taylor-Hood

# Initialisation:
t = 0.0; dumpn = 0; mn = 0
q = Function(Vq)
q.assign(q_)
q_file = File('prob1_test_outputs/prob1_adapt.pvd'.format(y=mn))
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
        H = construct_hessian(mesh, eta)
        if (remesh == 'y'):
            M = compute_steady_metric(mesh, H, eta, n2)
        else:
            M.interpolate(Expression([[n2, 0], [0, n2]]))

        # Adapt mesh and update FE setup:
        mesh = adapt(mesh, M)
        q_, q, u_, u, eta_, eta, b, Vq = update_SW_FE(mesh, u_, u, eta_, eta, b)

    # Solve weak problem:
    q_, q, u_, u, eta_, eta, q_solv = forward_linear_solver(q_, q, u_, eta_, b, Dt, Vq, params)

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
