from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

from utils import *

# Specify problem parameters:
dt = float(raw_input('Specify timestep (default 10): ') or 10.)
Dt = Constant(dt)
n = float(raw_input('Specify number of cells per m (default 5e-4): ') or 5e-4)
T = float(raw_input('Simulation duration in s (default 4200): ') or 4200.)
remesh = raw_input('Use adaptive meshing (y/n)?: ') or 'y'
if remesh == 'y':
    rm = int(raw_input('Timesteps per remesh (default 6)?: ') or 6)     # TODO: consider adaptive remeshing?
else:
    rm = int(T/dt)
    if remesh != 'n':
        raise ValueError('Please try again, typing y or n.')
ntype = raw_input('Normalisation type? (lp/manual): ') or 'lp'
ndump = 1

# Establish problem domain and variables:
mesh, Vq, q_, mu_, eta_, lam_, lm_, le_, b, BCs = tank_domain(n, test2d='y')
nx = int(4e5*n); ny = int(1e5*n)    # TODO: avoid this

# Set up forward problem solver:
q = Function(Vq)
q.assign(q_)
params = {'mat_type': 'matfree',
          'snes_type': 'ksponly',
          'pc_type': 'python',
          'pc_python_type': 'firedrake.AssembledPC',
          'assembled_pc_type': 'lu',
          'snes_lag_preconditioner': -1,
          'snes_lag_preconditioner_persists': True,}

q_, q, mu_, mu, eta_, eta, q_solv = SW_solve(q_, q, mu_, eta_, b, Dt, Vq, params, linear_form_2d)


# Initialise files and dump counter:
q_file = File('plots/adjoint_test_outputs/linear_forward.pvd')
t = 0.0
i = 0
dumpn = 0
q_file.write(mu, eta, time=t)

# Enter the forward timeloop:
while t < T - 0.5*dt:                       # TODO: implement adaptivity. Need remove SW_solve
    t += dt
    print 't = ', t, ' seconds'
    q_solv.solve()
    q_.assign(q)
    dumpn += 1
    if dumpn == ndump:
        dumpn -= ndump
        i += 1
        q_file.write(mu, eta, time=t)

print 'Forward problem solved.... now for the adjoint problem.'

# Set up adjoint weak problem:
lam = Function(Vq)
lam.assign(lam_)
lam_, lam, lm_, lm, le_, le, lam_solv = SW_solve(lam_, lam, lm_, le_, b, Dt, Vq, params, adj_linear_form_2d)
lm.rename('Adjoint fluid momentum')
le.rename('Adjoint free surface displacement')

# Initialise dump counter and files:
if dumpn == 0:
    dumpn = ndump
lam_file = File('plots/adjoint_test_outputs/linear_adjoint.pvd')
lam_file.write(lm, le, time=0)

# Enter the backward timeloop:
while t > 0:                               # TODO: implement adaptivity. Need remove SW_solve
    t -= dt
    print 't = ', t, ' seconds'
    lam_solv.solve()
    lam_.assign(lam)
    dumpn -= 1
    # Dump data:
    if dumpn == 0:
        dumpn += ndump
        i -= 1
        lam_file.write(lm, le, time=T-t)  # Note the time inversion
