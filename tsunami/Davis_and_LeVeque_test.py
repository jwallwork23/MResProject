from firedrake import *
import matplotlib.pyplot as plt
import numpy as np

############################ PARAMETERS ###############################

# Specify problem parameters:
dt = raw_input('Specify timestep (default 1): ') or 1
Dt = Constant(dt)
n = raw_input('Specify no. of cells per m (default 1e-3): ') or 1e-3
T = raw_input('Specify duration in s (default 4200): ') or 4200
tol = float(raw_input( \
    'Specify significance tolerance (default 0.1): ') or 0.1)
vid = raw_input('Show video output? (y/n, default n): ') or 'n'
if ((vid != 'y') & (vid != 'n')):
    raise ValueError('Please try again, choosing y or n.')
g = 9.81                    # Gravitational acceleration (m s^{-2})
ndump = 40

############################## FE SETUP ###############################

# Define domain and mesh:
lx = 4e5                    # 400 km ocean domain
nx = int(lx*n)
mesh = IntervalMesh(nx, lx)

# Define function spaces:
Vmu = FunctionSpace(mesh, 'CG', 2)      # \ Use Taylor-Hood elements
Ve = FunctionSpace(mesh, 'CG', 1)       # /
Vq = MixedFunctionSpace((Vmu, Ve))      # We have a mixed FE problem

# Construct functions to store forward and adjoint variables:
q_ = Function(Vq) 
lam_ = Function(Vq)
mu_, eta_ = q_.split()  # \ Split means we can interpolate the
lm_, le_ = lam_.split() # / initial condition into the two components

################# INITIAL CONDITIONS AND BATHYMETRY ###################

# Interpolate ICs:
mu_.interpolate(Expression(0.))
eta_.interpolate(Expression('(x[0] >= 1e5) & (x[0] <= 1.5e5) ? \
                             0.4*sin(pi*(x[0]-1e5)*2e-5) : 0.0'))

# Interpolate and plot bathymetry:
b = Function(Vq.sub(1), name = 'Bathymetry')
b.interpolate(Expression('x[0] <= 50000.0 ? -200.0 : -4000.0'))
plot(b)
plt.xlabel('Distance offshore (m)')
plt.ylabel('Bathymetry profile (m)')
plt.ylim([-5000.0, 0.0])
plt.savefig('tsunami_outputs/screenshots/bathy.png')
b.assign(-b)

###################### FORWARD WEAK PROBLEM ###########################

# Build the weak form of the timestepping algorithm, expressed as a 
# mixed nonlinear problem:
nu, ze = TestFunctions(Vq)
q = Function(Vq)
q.assign(q_)
mu, eta = split(q)       # \ Here split means we split up a function so
mu_, eta_ = split(q_)    # / it can be inserted into a UFL expression

# Establish forms (functions of the forward variable q), noting we only
# have a linear equation if the stong form is written in terms of a
# matrix:
L1 = (
    (ze * (eta-eta_) - Dt * mu * ze.dx(0) + \
    (mu-mu_) * nu + Dt * g * b * eta.dx(0) * nu) * dx
    )

# Set up the variational problem:
uprob1 = NonlinearVariationalProblem(L1, q)
usolver1 = NonlinearVariationalSolver(uprob1, solver_parameters={
                            'mat_type': 'matfree',
                            'snes_type': 'ksponly',
                            'pc_type': 'python',
                            'pc_python_type': 'firedrake.AssembledPC',
                            'assembled_pc_type': 'lu',
                            'snes_lag_preconditioner': -1, 
                            'snes_lag_preconditioner_persists': True,
                            })

# The function 'split' has two forms: now use the form which splits a 
# function in order to access its data:
mu_, eta_ = q_.split()
mu, eta = q.split()

# Store multiple functions:
mu.rename('Fluid momentum')
eta.rename('Free surface displacement')

######################## FORWARD TIMESTEPPING #########################

# Initialise time, counters and function arrays:
t = 0.0
i = 0
dumpn = 0
eta_snapshots = [Function(eta)]
eta_vid = [Function(eta)]
snaps = {0: 0.0, 1: 525.0, 2: 1365.0, 3: 2772.0, 4: 3255.0, 5: 4200.0}

# Initialise arrays for storage:
sig_eta = np.zeros((int(T*dt/ndump)+1, nx+1))       # \ Dimension
mu_vals = np.zeros((int(T*dt/ndump)+1, 2*nx+1))     # | pre-allocated
eta_vals = np.zeros((int(T*dt/ndump)+1, nx+1))      # / for speed
mu_vals[i,:] = mu.dat.data
eta_vals[i,:] = eta.dat.data

# Determine signifiant values (for domain of dependence plot):
for j in range(nx+1):
    if ((eta_vals[i,j] >= tol) | (eta_vals[i,j] <= -tol)):
        sig_eta[i,j] = 1

# Enter the forward timeloop:
while (t < T - 0.5*dt):
    t += dt
    print 't = ', t, ' seconds'
    usolver1.solve()
    q_.assign(q)
    dumpn += 1
    if (dumpn == ndump):
        dumpn -= ndump
        i += 1
        mu_vals[i,:] = mu.dat.data
        eta_vals[i,:] = eta.dat.data
        for j in range(nx+1):
            if ((eta_vals[i,j] >= tol) | (eta_vals[i,j] <= -tol)):
                sig_eta[i,j] = 1
        # Dump video data:
        if (vid == 'y'):
            eta_vid.append(Function(eta))
    # Dump snapshot data:
    if (t in snaps.values()):
        eta_snapshots.append(Function(eta))

print 'Forward problem solved.... now for the adjoint problem.'

######################## FORWARD PLOTTING #############################

for k in snaps:
    plot(eta_snapshots[k])
    plt.title('Surface at t = {y} seconds'.format(y=snaps[k]))
    plt.xlabel('Distance offshore (m)')
    plt.ylabel('Free surface displacement (m)')
    plt.ylim([-0.4, 0.5])
    plt.savefig('tsunami_outputs/screenshots/forward_t={y}.png'\
                .format(y=int(snaps[k])))

################### ADJOINT 'INITIAL' CONDITIONS ######################

# Interpolate final-time conditions:
lm_.interpolate(Expression(0.))
le_.interpolate(Expression('(x[0] >= 1e4) & (x[0] <= 2.5e4) ? \
                            0.4 : 0.0'))

###################### ADJOINT WEAK PROBLEM ###########################

# Establish test functions:
v, w = TestFunctions(Vq)
lam = Function(Vq)
lam.assign(lam_)
lm, le = split(lam)
lm_, le_ = split(lam_)

# Establish forms (functions of the adjoint variable lam):
L2 = (
    (w * (le-le_) + Dt * g * b * lm * w.dx(0) + \
    (lm-lm_) * v - Dt * le.dx(0) * v) * dx
    )

# Set up the variational problem:
uprob2 = NonlinearVariationalProblem(L2, lam)
usolver2 = NonlinearVariationalSolver(uprob2, solver_parameters={
                            'mat_type': 'matfree',
                            'snes_type': 'ksponly',
                            'pc_type': 'python',
                            'pc_python_type': 'firedrake.AssembledPC',
                            'assembled_pc_type': 'lu',
                            'snes_lag_preconditioner': -1, 
                            'snes_lag_preconditioner_persists': True,
                            })

# Split functions in order to access their data:
lm_, le_ = lam_.split()
lm, le = lam.split()

# Store multiple functions:
lm.rename('Adjoint fluid momentum')
le.rename('Adjoint free surface displacement')

######################## BACKWARD TIMESTEPPING ########################

# Initialise dump counter and function arrays:
if (dumpn == 0):
    dumpn = 10
le_snapshots = [Function(le)]
le_vid = [Function(le)]

# Initialise arrays for storage
sig_le = np.zeros((int(T*dt/ndump)+1, nx+1))        # \ Dimension
lm_vals = np.zeros((int(T*dt/ndump)+1, 2*nx+1))     # | pre-allocated
le_vals = np.zeros((int(T*dt/ndump)+1, nx+1))       # | for speed
q_dot_lam = np.zeros((int(T*dt/ndump)+1, nx+1))     # /
lm_vals[i,:] = lm.dat.data
le_vals[i,:] = le.dat.data

# Evaluate forward-adjoint inner products (noting mu and lm are in P2,
# while eta and le are in P1, so we need to evaluate at nodes):
q_dot_lam[i,:] = mu_vals[i,0::2] * lm_vals[i,0::2] + \
                 eta_vals[i,:] * le_vals[i,:]

# Determine significant values:
for j in range(nx+1):
    if ((le_vals[i,j] >= tol) | (le_vals[i,j] <= -tol)):
        sig_le[i,j] = 1
    if ((q_dot_lam[i,j] >= tol) | (q_dot_lam[i,j] <= -tol)):
        q_dot_lam[i,j] = 1
    else:
        q_dot_lam[i,j] = 0

# Enter the backward timeloop:
while (t > 0):
    t -= dt
    print 't = ', t, ' seconds'
    usolver2.solve()
    lam_.assign(lam)
    dumpn -= 1
    if (dumpn == 0):
        dumpn += ndump
        i -= 1
        lm_vals[i,:] = lm.dat.data
        le_vals[i,:] = le.dat.data
        q_dot_lam[i,:] = mu_vals[i,0::2] * lm_vals[i,0::2] + \
                         eta_vals[i,:] * le_vals[i,:]
        # Determine significant values:
        for j in range(nx+1):
            if ((le_vals[i,j] >= tol) | (le_vals[i,j] <= -tol)):
               sig_le[i,j] = 1
            if ((q_dot_lam[i,j] >= tol) | (q_dot_lam[i,j] <= -tol)):
                q_dot_lam[i,j] = 1
            else:
                q_dot_lam[i,j] = 0
        # Dump video data:
        if (vid == 'y'):
            le_vid.append(Function(le))
    # Dump snapshot data:
    if (t in snaps.values()):
        le_snapshots.append(Function(le))

######################## BACKWARD PLOTTING ############################

for k in snaps:
    plot(le_snapshots[k])
    plt.title('Surface at t = {y} seconds'.format(y=snaps[k]))
    plt.xlabel('Distance offshore (m)')
    plt.ylabel('Free surface displacement (m)')
    plt.ylim([-0.4, 0.5])
    plt.savefig('tsunami_outputs/screenshots/adjoint_t={y}.png' \
                .format(y=int(snaps[k])))

##plt.figure()
##fig, axes = plt.subplots(6, 2)
##k = 0
##for row in axes:
##    row[0].plot(eta_snapshots[k])
##    row[0].set_title('Surface at t = {y} seconds'.format(y=snaps[k]))
##    row[1].plot(le_snapshots[k])
##    row[1].set_title('Surface at t = {y} seconds'.format(y=snaps[k]))
##    plt.xlabel('Distance offshore (m)')
##    plt.ylabel('Free surface displacement (m)')
##    plt.ylim([-0.4, 0.5])
##    k += 1
##plt.savefig('tsunami_outputs/screenshots/subplots.png')

plt.clf()
plt.pcolor(sig_eta)
plt.title('Forward problem')
plt.xlabel('Distance offshore (m)')
plt.ylabel('Free surface displacement (m)')
plt.axis([0, nx+1, 0, int(T*dt/ndump)])
fig.savefig('tsunami_outputs/screenshots/significant_forward.png')

plt.clf()
plt.pcolor(sig_le)
plt.title('Adjoint problem')
plt.xlabel('Distance offshore (m)')
plt.ylabel('Free surface displacement (m)')
plt.axis([0, nx+1, 0, int(T*dt/ndump)])
fig.savefig('tsunami_outputs/screenshots/significant_adjoint.png')

plt.clf()
plt.pcolor(q_dot_lam)
plt.title('Domain of dependence')
plt.xlabel('Distance offshore (m)')
plt.ylabel('Forward-adjoint inner product (m^2)')
plt.axis([0, nx+1, 0, int(T*dt/ndump)])
fig.savefig('tsunami_outputs/screenshots/domain_of_dependence.png')

######################### VIDEO PLOTTING ##############################

if (vid == 'y'):
    plot(eta_vid)
    plt.xlabel('Distance offshore (m)')
    plt.ylabel('Free surface displacement (m)')
    plt.ylim([-0.4, 0.5])
    plt.show()

if (vid == 'y'):
    plot(le_vid)
    plt.xlabel('Distance offshore (m)')
    plt.ylabel('Free surface displacement (m)')
    plt.ylim([-0.4, 0.5])
    plt.show()
