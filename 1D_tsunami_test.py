from firedrake import *
import numpy as np
from matplotlib import rc
from time import clock
from utils import domain_1d
import matplotlib
matplotlib.use('TkAgg')             # Change backend to resolve framework problems
import matplotlib.pyplot as plt

print ''
print '******************************** 1D TSUNAMI TEST ********************************'
print ''

# Specify problem parameters:
dt = float(raw_input('Specify timestep (default 1): ') or 1.)
Dt = Constant(dt)
tol = float(raw_input('Specify significance tolerance (default 0.05): ') or 0.05)
vid = raw_input('Show video output? (y/n, default n): ') or 'n'
if vid not in ('y', 'n'):
    raise ValueError('Please try again, choosing y or n.')
n = 1e-3        # Number of cells per km
T = 4200.       # Simulation duration (s)
ndump = 60      # Timesteps per data dump
g = 9.81        # Gravitational acceleration (m s^{-2})
print ''

# Check CFL criterion is satisfied for this discretisation:
assert(dt < 1. / (n * np.sqrt(g * 4000.)))                                      # Maximal wavespeed sqrt(gb)

# Begin timing:
tic1 = clock()

# Establish problem domain and variables:
mesh, Vq, q_, mu_, eta_, lam_, lm_, le_, b = domain_1d(n)
nx = int(4e5 * n)                                                               # For data access purposes
coords = mesh.coordinates.dat.data

# Set up functions of the forward weak problem:
q = Function(Vq)
q.assign(q_)
nu, ze = TestFunctions(Vq)
mu, eta = split(q)
mu_, eta_ = split(q_)

# For timestepping we consider the implicit midpoint rule and so must create new 'mid-step' functions:
muh = 0.5 * (mu + mu_)
etah = 0.5 * (eta + eta_)

# Specify solution parameters:
params = {'mat_type': 'matfree',
          'snes_type': 'ksponly',
          'pc_type': 'python',
          'pc_python_type': 'firedrake.AssembledPC',
          'assembled_pc_type': 'lu',
          'snes_lag_preconditioner': -1,
          'snes_lag_preconditioner_persists': True}

# Set up the variational problem:
L = ((eta - eta_) * ze - Dt * muh * ze.dx(0) + (mu - mu_) * nu + Dt * g * b * etah.dx(0) * nu) * dx
q_prob = NonlinearVariationalProblem(L, q)
q_solv = NonlinearVariationalSolver(q_prob, solver_parameters=params)

# 'Split' functions in order to access their data and then relabel:
mu_, eta_ = q_.split()
mu, eta = q.split()
mu.rename('Fluid momentum')
eta.rename('Free surface displacement')

# Initialise time, counters and function arrays:
t = 0.0
i = 0
dumpn = 0
eta_snapshots = [Function(eta)]
eta_vid = [Function(eta)]
snaps = {0: 0.0, 1: 525.0, 2: 1365.0, 3: 2772.0, 4: 3255.0, 5: 4200.0}

# Initialise arrays for storage (with dimensions pre-allocated for speed):
sig_eta = np.zeros((int(T / (ndump * dt)) + 1, nx + 1))
mu_vals = np.zeros((int(T / (ndump * dt)) + 1, nx + 1))
eta_vals = np.zeros((int(T / (ndump * dt)) + 1, nx + 1))
m = np.zeros((int(T / (ndump * dt)) + 1))
mu_vals[i, :] = mu.at(coords, dont_raise=True)  # Interpolated at vertices to be consistent with P1 free surface field
eta_vals[i, :] = eta.dat.data
m[i] = np.log2(max(eta_vals[i, 0], 0.5))

# Determine signifiant values (for domain of dependence plot):
for j in range(nx + 1):
    if (eta_vals[i, j] >= tol) | (eta_vals[i, j] <= -tol):
        sig_eta[i, j] = 1

# Enter the forward timeloop:
print '******************************** Forward solver ********************************'
print ''
while t < T - 0.5 * dt:
    t += dt
    print 't = ', t, ' seconds'
    q_solv.solve()
    q_.assign(q)
    dumpn += 1
    if dumpn == ndump:
        dumpn -= ndump
        i += 1
        mu_vals[i, :] = mu.at(coords, dont_raise=True)
        eta_vals[i, :] = eta.dat.data

        # Determine significant values:
        for j in range(nx + 1):
            if (eta_vals[i, j] >= tol) | (eta_vals[i, j] <= - tol):
                sig_eta[i, j] = 1
                
        # Implement damage measures:
        m[i] = np.log2(max(eta_vals[i, 0], 0.5))
        
        # Dump video data:
        if vid == 'y':
            eta_vid.append(Function(eta))
            
    # Dump snapshot data:
    if t in snaps.values():
        eta_snapshots.append(Function(eta))

print '... forward problem solved...'
print ''

# Set up functions of the adjoint weak problem:
lam = Function(Vq)
lam.assign(lam_)
v, w = TestFunctions(Vq)
lm, le = split(lam)
lm_, le_ = split(lam_)

# Create 'mid-step' functions:
lmh = 0.5 * (lm + lm_)
leh = 0.5 * (le + le_)

# Set up the variational problem
L = ((le - le_) * w + Dt * g * b * lmh * w.dx(0) + (lm - lm_) * v - Dt * leh.dx(0) * v) * dx
lam_prob = NonlinearVariationalProblem(L, lam)
lam_solv = NonlinearVariationalSolver(lam_prob, solver_parameters=params)

# 'Split' functions to access their data and then relabel:
lm_, le_ = lam_.split()
lm, le = lam.split()
lm.rename('Adjoint fluid momentum')
le.rename('Adjoint free surface displacement')

# Initialise dump counter and function arrays:
if dumpn == 0:
    dumpn = ndump
le_snapshots = [Function(le)]
le_vid = [Function(le)]

# Initialise arrays for storage (with dimensions pre-allocated for speed):
sig_le = np.zeros((int(T / (ndump * dt)) + 1, nx + 1))
lm_vals = np.zeros((int(T / (ndump * dt)) + 1, nx + 1))
le_vals = np.zeros((int(T / (ndump * dt)) + 1, nx + 1))
q_dot_lam = np.zeros((int(T / (ndump * dt)) + 1, nx + 1))
lm_vals[i, :] = lm.at(coords, dont_raise=True)
le_vals[i, :] = le.dat.data

# Evaluate forward-adjoint inner products:
q_dot_lam[i, :] = mu_vals[i, :] * lm_vals[i, :] + eta_vals[i, :] * le_vals[i, :]

# Determine significant values:
for j in range(nx + 1):
    if (le_vals[i, j] >= tol) | (le_vals[i, j] <= - tol):
        sig_le[i, j] = 1
    if (q_dot_lam[i, j] >= tol) | (q_dot_lam[i, j] <= - tol):
        q_dot_lam[i, j] = 1
    else:
        q_dot_lam[i, j] = 0

# Enter the backward timeloop:
print '******************************** Adjoint solver ********************************'
print ''
while t > 0.5 * dt:
    t -= dt
    print 't = ', t, ' seconds'
    lam_solv.solve()
    lam_.assign(lam)
    dumpn -= 1
    if dumpn == 0:
        dumpn += ndump
        i -= 1
        lm_vals[i, :] = lm.at(coords, dont_raise=True)
        le_vals[i, :] = le.dat.data
        q_dot_lam[i, :] = mu_vals[i, :] * lm_vals[i, :] + eta_vals[i, :] * le_vals[i, :]

        # Determine significant values:
        for j in range(nx + 1):
            if (le_vals[i, j] >= tol) | (le_vals[i, j] <= - tol):
                sig_le[i, j] = 1
            if (q_dot_lam[i, j] >= tol) | (q_dot_lam[i, j] <= - tol):
                q_dot_lam[i, j] = 1
            else:
                q_dot_lam[i, j] = 0

        # Dump video data:
        if vid == 'y':
            le_vid.append(Function(le))

    # Dump snapshot data:
    if t in snaps.values():
        le_snapshots.append(Function(le))
print '... adjoint problem solved... just need to plot results.'

# Font formatting:
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

if vid == 'y':
    # Plot solution videos:
    for k in (eta_vid, le_vid):
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plot(k)
        plt.axvline(5e4, linestyle='--', color='black')
        plt.xlabel(r'Distance offshore (m)')
        plt.ylabel(r'Free surface displacement (m)')
        plt.ylim([-0.4, 0.5])
        plt.xlim(plt.xlim()[::-1])
        plt.show()
else:
    # Plot bathymetry profile:
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    b.assign(-b)
    plot(b)
    plt.axvline(5e4, linestyle='--', color='black')
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.xlabel(r'Distance offshore (m)')
    plt.ylabel(r'Bathymetry profile (m)')
    plt.ylim([-5000.0, 0.0])
    plt.xlim(plt.xlim()[::-1])
    plt.savefig('plots/tsunami_outputs/screenshots/bathy.png')

    for k in snaps:
        # Plot forward solutions:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plot(eta_snapshots[k])
        plt.axvline(5e4, linestyle='--', color='black')
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.title(r'Surface at t = {y} seconds'.format(y=snaps[k]))
        plt.xlabel(r'Distance offshore (m)')
        plt.ylabel(r'Free surface displacement (m)')
        plt.ylim([-0.4, 0.5])
        plt.xlim(plt.xlim()[::-1])
        plt.savefig('plots/tsunami_outputs/screenshots/forward_t={y}.png'.format(y=int(snaps[k])))
        # Plot adjoint solutions:
        plt.clf()
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plot(le_snapshots[k])
        plt.axvline(5e4, linestyle=':', color='black')
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.title(r'Surface at t = {y} seconds'.format(y=snaps[k]))
        plt.xlabel(r'Distance offshore (m)')
        plt.ylabel(r'Free surface displacement (m)')
        plt.ylim([-0.4, 0.5])
        plt.xlim(plt.xlim()[::-1])
        plt.savefig('plots/tsunami_outputs/screenshots/adjoint_t={y}.png'.format(y=int(snaps[k])))

    plots = {'Forward': sig_eta, 'Adjoint': sig_le, 'Domain of dependence': q_dot_lam}

    # Make significance and domain-of-dependence plots:
    for k in plots:
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.pcolor(plots[k], cmap='gray_r')
        plt.axvline(50, linestyle='--', color='black')
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.xlabel(r'Distance offshore (km)')
        plt.axis([0, nx + 1, 0, int(T / (ndump * dt))])
        plt.xlim(plt.xlim()[::-1])
        plt.ylabel(r'Time (mins)')
        if k == 'Domain of dependence':
            plt.title(r'{y}'.format(y=k))
            plt.savefig('plots/tsunami_outputs/screenshots/domain_of_dependence.png')
        else:
            plt.title(r'{y} problem'.format(y=k))
            plt.savefig('plots/tsunami_outputs/screenshots/sig_{y}.png'.format(y=k))

    # Plot damage measures:
    plt.clf()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.plot(range(0, int(T) + 1, int(dt * ndump)), m, color='black')
    plt.axis([0, T, -1.5, 3.5])
    plt.axhline(-1, linestyle='--', color='blue')
    plt.axhline(0, linestyle='--', color='green')
    plt.axhline(1, linestyle='--', color='yellow')
    plt.axhline(2, linestyle='--', color='orange')
    plt.axhline(3, linestyle='--', color='red')
    plt.annotate('Severe damage', xy=(0.7 * T, 3), xytext=(0.72 * T, 3.2),
                 arrowprops=dict(facecolor='red', shrink=0.05))
    plt.annotate('Some inland damage', xy=(0.7 * T, 2), xytext=(0.72 * T, 2.2),
                 arrowprops=dict(facecolor='orange', shrink=0.05))
    plt.annotate('Shore damage', xy=(0.7 * T, 1), xytext=(0.72 * T, 1.2),
                 arrowprops=dict(facecolor='yellow', shrink=0.05))
    plt.annotate('Very little damage', xy=(0.7 * T, 0), xytext=(0.72 * T, 0.2),
                 arrowprops=dict(facecolor='green', shrink=0.05))
    plt.annotate('No damage', xy=(0.7 * T, -1), xytext=(0.72 * T, -0.8),
                 arrowprops=dict(facecolor='blue', shrink=0.05))
    plt.xlabel(r'Time (s)')
    plt.ylabel(r'm (dimensionless)')
    plt.title(r'Damage measures')
    plt.savefig('plots/tsunami_outputs/screenshots/1Ddamage_measures.png')
