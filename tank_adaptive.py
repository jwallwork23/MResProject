from firedrake import *
import numpy as np

from utils import *

# Specify problem parameters:
mode = raw_input('Linear or nonlinear equations? (l/n): ') or 'l'
if (mode != 'l') & (mode != 'n'):
    raise ValueError('Please try again, choosing l or n.')
bathy = raw_input('Non-trivial bathymetry? (y/n): ') or 'n'
if (bathy != 'y') & (bathy != 'n'):
    raise ValueError('Please try again, choosing y or n.')
dt = float(raw_input('Timestep (default 0.1)?: ') or 0.1)               # TODO: consider adaptive timestepping?
Dt = Constant(dt)
n = int(raw_input('Mesh cells per m (default 16)?: ') or 16)
T = float(raw_input('Simulation duration in s (default 5)?: ') or 5.0)
remesh = raw_input('Use adaptive meshing (y/n)?: ') or 'y'
if remesh == 'y':
    rm = int(raw_input('Timesteps per remesh (default 6)?: ') or 6)     # TODO: consider adaptive remeshing?
else:
    rm = int(T/dt)
    if remesh != 'n':
        raise ValueError('Please try again, typing y or n.')
ntype = raw_input('Normalisation type? (lp/manual): ') or 'lp'
ndump = 1

# Establish tank domain
mesh, Vq, q_, u_, eta_, lam_, lu_, le_, b, BCs = tank_domain(n, bath=bathy)

# Initialisation:
t = 0.0
dumpn = 0
mn = 0
q = Function(Vq)
q.assign(q_)
##q_file = File('plots/prob1_test_outputs/prob1_adapt.pvd')
params = {'mat_type': 'matfree',
          'snes_type': 'ksponly',
          'pc_type': 'python',
          'pc_python_type': 'firedrake.AssembledPC',
          'assembled_pc_type': 'lu',
          'snes_lag_preconditioner': -1,
          'snes_lag_preconditioner_persists': True,}
if mode == 'l':
    form = linear_form
else:
    form = nonlinear_form

# Set up functions of weak problem:
v, ze = TestFunctions(Vq)
u, eta = split(q)
u_, eta_ = split(q_)

while t < T-0.5*dt:

    mn += 1
    step_file = File('plots/prob1_test_outputs/prob1_adapt_step_{y}.pvd'.format(y=mn))  # TODO: get rid of this

    if t == 0.0:

        # Establish form:
        L = form(u, u_, eta, eta_, v, ze, b, Dt, n)

        # Set up the variational problem
        q_prob = NonlinearVariationalProblem(L, q)
        q_solv = NonlinearVariationalSolver(q_prob, solver_parameters=params)

        # The function 'split' has two forms: now use the form which splits a function in order to access its data
        u_, eta_ = q_.split()
        u, eta = q.split()

        # Store multiple functions
        u.rename('Fluid velocity')
        eta.rename('Free surface displacement')

    else:                                                           # TODO: Could adapt straight away?

        # Set up metric:
        Vm = TensorFunctionSpace(mesh, 'CG', 1)
        M = Function(Vm)

        # Build Hessian and (hence) metric:
        H = construct_hessian(mesh, Vm, eta)
        if remesh == 'y':
            M = compute_steady_metric(mesh, Vm, H, eta, normalise=ntype)
        else:
            M.interpolate(Expression([[n*n, 0], [0, n*n]]))

        # Adapt mesh and update FE setup:
        mesh_ = mesh
        mesh = adapt(mesh, M)
        q_, q, u_, u, eta_, eta, b, Vq = update_SW_FE(mesh_, mesh, u_, u, eta_, eta, b)

        # Set up functions of weak problem:
        v, ze = TestFunctions(Vq)
        u, eta = split(q)
        u_, eta_ = split(q_)

        # Establish form:
        L = form(u, u_, eta, eta_, v, ze, b, Dt, n)

        # Set up the variational problem
        q_prob = NonlinearVariationalProblem(L, q)
        q_solv = NonlinearVariationalSolver(q_prob, solver_parameters=params)

        # The function 'split' has two forms: now use the form which splits a function in order to access its data
        u_, eta_ = q_.split()
        u, eta = q.split()


    if t == 0.0:
##        q_file.write(u, eta, time=t)
        step_file.write(u, eta, time=t)                                 # TODO: get rid of this
    cnt = 0

    # Enter the inner timeloop:
    while cnt < rm:
        t += dt
        print 't = ', t, ' seconds, mesh number = ', mn
        cnt += 1
        q_solv.solve()
        q_.assign(q)
        dumpn += 1
        if dumpn == ndump:
            dumpn -= ndump
##            q_file.write(u, eta, time=t)
            step_file.write(u, eta, time=t)                             # TODO: get rid of this
