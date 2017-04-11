from firedrake import *
import matplotlib.pyplot as plt
import numpy as np

############################ PARAMETERS ###############################

# Specify problem parameters:
dt = raw_input('Specify timestep (default 1): ') or 1
Dt = Constant(dt)
n = raw_input('Specify no. of cells per m (default 1e-3): ') or 1e-3
T = raw_input('Specify duration in s (default 4200): ') or 4200
tol = raw_input('Specify significant tolerance (default 0.1): ') or 0.1
vid = raw_input('Video output? (y/n, default n): ') or 'n'
if ((vid != 'y') & (vid != 'n')):
    raise ValueError('Please try again, choosing y or n.')
g = 9.81    # Gravitational acceleration
ndump = 40

############################## FE SETUP ###############################

# Define domain and mesh
lx = 4e5                    # 400 km ocean domain
nx = int(lx*n)
mesh = IntervalMesh(nx, lx)

# Define function spaces:
Vmu = FunctionSpace(mesh, 'CG', 2)      # \ Use Taylor-Hood elements
Ve = FunctionSpace(mesh, 'CG', 1)       # /
Vq = MixedFunctionSpace((Vmu, Ve))      # We have a mixed FE problem

# Construct functions to store forward problem variables:
q_ = Function(Vq)
mu_, eta_ = q_.split()

# Construct functions to store adjoint problem variables:
lam_ = Function(Vq)
lm_, le_ = lam_.split()

################# INITIAL CONDITIONS AND BATHYMETRY ###################

# Interpolate ICs:
mu_.interpolate(Expression(0.))
eta_.interpolate(Expression('(x[0] >= 1e5) & (x[0] <= 1.5e5) ? \
                             0.4*sin(pi*(x[0]-1e5)*2e-5) : 0.0'))

# Interpolate and plot bathymetry:
b = Function(Vq.sub(1), name = 'Bathymetry')
b.interpolate(Expression('x[0] <= 50000.0 ? -200.0 : -4000.0'))
fig1 = plt.figure(1)
plot(b)
plt.xlabel('Distance offshore (m)')
plt.ylabel('Bathymetry profile (m)')
plt.ylim([-5000.0, 0.0])
fig1.savefig('tsunami_outputs/screenshots/bathy.png')
plt.close(fig1)
b.assign(-b)

###################### FORWARD WEAK PROBLEM ###########################

# Build the weak form of the timestepping algorithm, expressed as a 
# mixed nonlinear problem
nu, ze = TestFunctions(Vq)
q = Function(Vq)
q.assign(q_)
mu, eta = split(q)       # \ Here split means we split up a function so
mu_, eta_ = split(q_)    # / it can be inserted into a UFL expression

# Establish forms (functions of the output q), noting we only have a 
# linear equation if the stong form is written in terms of a matrix:
L1 = (
    (ze * (eta-eta_) - Dt * mu * ze.dx(0) + \
    (mu-mu_) * nu + Dt * g * b * eta.dx(0) * nu) * dx
    )

# Set up the problem
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
# function in order to access its data
mu_, eta_ = q_.split()
mu, eta = q.split()

# Store multiple functions:
mu.rename('Fluid momentum')
eta.rename('Free surface displacement')

######################## FORWARD TIMESTEPPING #########################

# Initialise time, arrays and dump counter:
t = 0.0
dumpn = 0
eta_snapshots = [Function(eta)]
eta_vid = [Function(eta)]

# Establish a function and array to indicate significant eta values:
s = Function(Vq.sub(1)) 
s.assign(conditional(lt(eta, tol), 0.0, 1.0))       # TODO : use abs
sig_eta = np.zeros((int(T*dt/ndump)+1, nx+1))       # \ Dimension
mu_vals = np.zeros((int(T*dt/ndump)+1, 2*nx+1))     # | pre-allocated
eta_vals = np.zeros((int(T*dt/ndump)+1, nx+1))      # / for speed
i = 0
sig_eta[i,:] = s.dat.data
mu_vals[i,:] = mu.dat.data
eta_vals[i,:] = eta.dat.data

# Enter the timeloop:
while (t < T - 0.5*dt):
    t += dt
    print 't = ', t, ' seconds'
    usolver1.solve()
    q_.assign(q)
    dumpn += 1
    if (dumpn == ndump):
        dumpn -= ndump
        i += 1
        s.assign(conditional(lt(eta, tol), 0.0, 1.0))
        sig_eta[i,:] = s.dat.data
        mu_vals[i,:] = mu.dat.data
        eta_vals[i,:] = eta.dat.data
        # Dump video data:
        if (vid == 'y'):
            eta_vid.append(Function(eta))
    # Dump snapshot data:
    if (t in (525.0, 1365.0, 2772.0, 3655.0, 4200.0)):
        eta_snapshots.append(Function(eta))

print 'Forward problem solved.... now for the adjoint problem.'

######################## FORWARD PLOTTING #############################

fig2 = plt.figure(2)
plot(eta_snapshots[0])
plt.title('Surface at t = 0 seconds')
plt.xlabel('Distance offshore (m)')
plt.ylabel('Free surface displacement (m)')
plt.ylim([-0.4, 0.5])
fig2.savefig('tsunami_outputs/screenshots/forward_t=0.png')
plt.close(fig2)

fig3 = plt.figure(3)
plot(eta_snapshots[1])
plt.title('Surface at t = 525 seconds')
plt.xlabel('Distance offshore (m)')
plt.ylabel('Free surface displacement (m)')
plt.ylim([-0.4, 0.5])
fig3.savefig('tsunami_outputs/screenshots/forward_t=525.png')
plt.close(fig3)

fig4 = plt.figure(4)
plot(eta_snapshots[2])
plt.title('Surface at t = 1365 seconds')
plt.xlabel('Distance offshore (m)')
plt.ylabel('Free surface displacement (m)')
plt.ylim([-0.4, 0.5])
fig4.savefig('tsunami_outputs/screenshots/forward_t=1365.png')
plt.close(fig4)

fig5 = plt.figure(5)
plot(eta_snapshots[3])
plt.title('Surface at t = 2772 seconds')
plt.xlabel('Distance offshore (m)')
plt.ylabel('Free surface displacement (m)')
plt.ylim([-0.4, 0.5])
fig5.savefig('tsunami_outputs/screenshots/forward_t=2772.png')
plt.close(fig5)

fig6 = plt.figure(6)
plot(eta_snapshots[4])
plt.title('Surface at t = 3255 seconds')
plt.xlabel('Distance offshore (m)')
plt.ylabel('Free surface displacement (m)')
plt.ylim([-0.4, 0.5])
fig6.savefig('tsunami_outputs/screenshots/forward_t=3255.png')
plt.close(fig6)

fig7 = plt.figure(7)
plot(eta_snapshots[5])
plt.title('Surface at t = 4200 seconds')
plt.xlabel('Distance offshore (m)')
plt.ylabel('Free surface displacement (m)')
plt.ylim([-0.4, 0.5])
fig7.savefig('tsunami_outputs/screenshots/forward_t=4200.png')
plt.close(fig7)

################### ADJOINT 'INITIAL' CONDITIONS ######################

# Interpolate final-time conditions:
lm_.interpolate(Expression(0.))
le_.interpolate(Expression('(x[0] >= 1e4) & (x[0] <= 2.5e4) ? \
                            0.4 : 0.0'))

###################### ADJOINT WEAK PROBLEM ###########################

# Build the weak form of the timestepping algorithm, expressed as a 
# mixed nonlinear problem
v, w = TestFunctions(Vq)
lam = Function(Vq)
lam.assign(lam_)
lm, le = split(lam)       # \ Here split means we split up a function so
lm_, le_ = split(lam_)    # / it can be inserted into a UFL expression

# Establish forms (functions of the output q), noting we only have a 
# linear equation if the stong form is written in terms of a matrix:
L2 = (
    (w * (le-le_) + Dt * g * b * lm * w.dx(0) + \
    (lm-lm_) * v - Dt * le.dx(0) * v) * dx
    )

# Set up the problem
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

# The function 'split' has two forms: now use the form which splits a 
# function in order to access its data
lm_, le_ = lam_.split()
lm, le = lam.split()

# Store multiple functions:
lm.rename('Adjoint fluid momentum')
le.rename('Adjoint free surface displacement')

######################## BACKWARD TIMESTEPPING ########################

# Initialise arrays:
le_snapshots = [Function(le)]
le_vid = [Function(le)]
if (dumpn == 0):
    dumpn = 10
s.assign(conditional(lt(le, tol), 0.0, 1.0))        # TODO : use abs
sig_le = np.zeros((int(T*dt/ndump)+1, nx+1))        # \ Dimension
lm_vals = np.zeros((int(T*dt/ndump)+1, 2*nx+1))     # | pre-allocated
le_vals = np.zeros((int(T*dt/ndump)+1, nx+1))       # | for speed
q_dot_lam = np.zeros((int(T*dt/ndump)+1, nx+1))     # /
sig_le[i,:] = s.dat.data
lm_vals[i,:] = mu.dat.data
le_vals[i,:] = eta.dat.data
q_dot_lam[i,:] = mu_vals[i,0::2] * lm_vals[i,0::2] + \
                 eta_vals[i,:] * le_vals[i,:]

# Enter the timeloop:
while (t > 0):
    t -= dt
    print 't = ', t, ' seconds'
    usolver2.solve()
    lam_.assign(lam)
    dumpn -= 1
    if (dumpn == 0):
        dumpn += ndump
        i -= 1
        s.assign(conditional(lt(le, tol), 0.0, 1.0))
        sig_le[i,:] = s.dat.data
        lm_vals[i,:] = mu.dat.data
        le_vals[i,:] = eta.dat.data
        # Evaluate forward-adjoint inner products (noting mu is in
        # CG2, while eta is in CG1, so we need to evaluate at nodes):
        q_dot_lam[i,:] = mu_vals[i,0::2] * lm_vals[i,0::2] + \
                         eta_vals[i,:] * le_vals[i,:]
        # Dump video data:
        if (vid == 'y'):
            le_vid.append(Function(le))
    # Dump snapshot data:
    if (t in (0.0, 525.0, 1365.0, 2772.0, 3655.0)):
        le_snapshots.append(Function(le))

######################## BACKWARD PLOTTING ############################

fig8 = plt.figure(8)
plot(le_snapshots[0])
plt.title('Adjoint at t = 4200 seconds')
plt.xlabel('Distance offshore (m)')
plt.ylabel('Free surface displacement (m)')
plt.ylim([-0.4, 0.5])
fig8.savefig('tsunami_outputs/screenshots/adjoint_t=4200.png')
plt.close(fig8)

fig9 = plt.figure(9)
plot(le_snapshots[1])
plt.title('Adjoint at t = 3255 seconds')
plt.xlabel('Distance offshore (m)')
plt.ylabel('Free surface displacement (m)')
plt.ylim([-0.4, 0.5])
fig9.savefig('tsunami_outputs/screenshots/adjoint_t=3255.png')
plt.close(fig9)

fig10 = plt.figure(10)
plot(le_snapshots[2])
plt.title('Adjoint at t = 2772 seconds')
plt.xlabel('Distance offshore (m)')
plt.ylabel('Free surface displacement (m)')
plt.ylim([-0.4, 0.5])
fig10.savefig('tsunami_outputs/screenshots/adjoint_t=2772.png')
plt.close(fig10)

fig11 = plt.figure(11)
plot(le_snapshots[3])
plt.title('Adjoint at t = 1365 seconds')
plt.xlabel('Distance offshore (m)')
plt.ylabel('Free surface displacement (m)')
plt.ylim([-0.4, 0.5])
fig11.savefig('tsunami_outputs/screenshots/adjoint_t=1365.png')
plt.close(fig11)

fig12 = plt.figure(12)
plot(le_snapshots[4])
plt.title('Adjoint at t = 525 seconds')
plt.xlabel('Distance offshore (m)')
plt.ylabel('Free surface displacement (m)')
plt.ylim([-0.4, 0.5])
fig12.savefig('tsunami_outputs/screenshots/adjoint_t=525.png')
plt.close(fig12)

fig13 = plt.figure(13)
plot(le_snapshots[5])
plt.title('Adjoint at t = 0 seconds')
plt.xlabel('Distance offshore (m)')
plt.ylabel('Free surface displacement (m)')
plt.ylim([-0.4, 0.5])
fig13.savefig('tsunami_outputs/screenshots/adjoint_t=0.png')
plt.close(fig13)

fig14 = plt.figure(14)
plt.pcolor(sig_eta)
plt.title('Forward problem')
plt.xlabel('Distance offshore (m)')
plt.ylabel('Free surface displacement (m)')
plt.axis([0, nx+1, 0, int(T*dt/ndump)])
plt.show()
fig14.savefig('tsunami_outputs/screenshots/significant_forward.png')
plt.close(fig14)

fig15 = plt.figure(15)
plt.pcolor(sig_le)
plt.title('Adjoint problem')
plt.xlabel('Distance offshore (m)')
plt.ylabel('Free surface displacement (m)')
plt.axis([0, nx+1, 0, int(T*dt/ndump)])
plt.show()
fig15.savefig('tsunami_outputs/screenshots/significant_adjoint.png')
plt.close(fig15)

# TODO : Make this into a significance plot
fig16 = plt.figure(16)
plt.pcolor(q_dot_lam)
plt.title('Domain of dependence')
plt.xlabel('Distance offshore (m)')
plt.ylabel('Forward-adjoint inner product (m^2)')
plt.axis([0, nx+1, 0, int(T*dt/ndump)])
plt.show()
fig16.savefig('tsunami_outputs/screenshots/domain_of_dependence.png')
plt.close(fig16)

######################### VIDEO PLOTTING ##############################

plt.figure(17)
if (vid == 'y'):
    plot(eta_vid)
    plt.xlabel('Distance offshore (m)')
    plt.ylabel('Free surface displacement (m)')
    plt.ylim([-0.4, 0.5])
    plt.show()

plt.figure(18)
if (vid == 'y'):
    plot(le_vid)
    plt.xlabel('Distance offshore (m)')
    plt.ylabel('Free surface displacement (m)')
    plt.ylim([-0.4, 0.5])
    plt.show()
