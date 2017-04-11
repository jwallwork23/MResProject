from firedrake import *
import matplotlib.pyplot as plt

############################ PARAMETERS ###############################

# Specify problem parameters:
dt = raw_input('Specify timestep (default 1): ') or 1
Dt = Constant(dt)
n = raw_input('Specify no. of cells per m (default 1e-3): ') or 1e-3
T = raw_input('Specify duration in s (default 4200): ') or 4200
g = 9.81    # Gravitational acceleration
vid = raw_input('Video output? (y/n, default n)') or 'n'
if ((vid != 'l') & (vid != 'n')):
    raise ValueError('Please try again, choosing l or n.')
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
eta_.interpolate(Expression( '(x[0] >= 1e5) & (x[0] <= 1.5e5) ? \
                             0.4*sin(pi*(x[0]-1e5)*2e-5) : 0.0'))

# Interpolate bathymetry:
b = Function(Vq.sub(1), name = 'Bathymetry')
b.interpolate(Expression('x[0] <= 50000.0 ? -200.0 : -4000.0'))
plot(b)
plt.xlabel('Location in ocean domain (m)')
plt.ylabel('Bathymetry profile (m)')
plt.ylim([-5000.0, 0.0])
plt.show()
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
snapshots1 = [Function(eta)]
video1 = [Function(eta)]

# Enter the timeloop:
while (t < T - 0.5*dt):
    t += dt
    print 't = ', t, ' seconds'
    usolver1.solve()
    q_.assign(q)
    dumpn += 1
    # Dump video data:
    if ((vid == 'y') & (dumpn == ndump)):
        dumpn -= ndump
        video1.append(Function(eta))
    # Dump snapshot data:
    if (t in (525.0, 1365.0, 2772.0, 3655.0, 4200.0)):
        snapshots1.append(Function(eta))

######################## FORWARD PLOTTING #############################

plot(snapshots1[0])
plt.title('Surface at t = 0 seconds')
plt.xlabel('Location in ocean domain (m)')
plt.ylabel('Free surface displacement (m)')
plt.show()

plot(snapshots1[1])
plt.title('Surface at t = 525 seconds')
plt.xlabel('Location in ocean domain (m)')
plt.ylabel('Free surface displacement (m)')
plt.show()

plot(snapshots1[2])
plt.title('Surface at t = 1365 seconds')
plt.xlabel('Location in ocean domain (m)')
plt.ylabel('Free surface displacement (m)')
plt.show()

plot(snapshots1[3])
plt.title('Surface at t = 2772 seconds')
plt.xlabel('Location in ocean domain (m)')
plt.ylabel('Free surface displacement (m)')
plt.show()

plot(snapshots1[4])
plt.title('Surface at t = 3255 seconds')
plt.xlabel('Location in ocean domain (m)')
plt.ylabel('Free surface displacement (m)')
plt.show()

plot(snapshots1[5])
plt.title('Surface at t = 4200 seconds')
plt.xlabel('Location in ocean domain (m)')
plt.ylabel('Free surface displacement (m)')
plt.show()

if (vid == 'y'):
    plot(video1)
    plt.xlabel('Location in ocean domain (m)')
    plt.ylabel('Free surface displacement (m)')
    plt.show()

################### ADJOINT 'INITIAL' CONDITIONS ######################

# Interpolate final-time conditions:
lm_.interpolate(Expression(0.))
le_.interpolate(Expression( '(x[0] >= 1e4) & (x[0] <= 2.5e4) ? \
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
snapshots2 = [Function(le)]
video2 = [Function(le)]

# Enter the timeloop:
while (t > 0):
    t -= dt
    print 't = ', t, ' seconds'
    usolver2.solve()
    lam_.assign(lam)
    dumpn -= 1
    # Dump video data:
    if ((vid == 'y') & (dumpn == 0)):
        dumpn += ndump
        video2.append(Function(le))
    # Dump snapshot data:
    if (t in (0.0, 525.0, 1365.0, 2772.0, 3655.0)):
        snapshots2.append(Function(le))

######################## BACKWARD PLOTTING ############################

plot(snapshots2[0])
plt.title('Adjoint at t = 4200 seconds')
plt.xlabel('Location in ocean domain (m)')
plt.ylabel('Free surface displacement (m)')
plt.show()

plot(snapshots2[1])
plt.title('Adjoint at t = 3255 seconds')
plt.xlabel('Location in ocean domain (m)')
plt.ylabel('Free surface displacement (m)')
plt.show()

plot(snapshots2[2])
plt.title('Adjoint at t = 2772 seconds')
plt.xlabel('Location in ocean domain (m)')
plt.ylabel('Free surface displacement (m)')
plt.show()

plot(snapshots2[3])
plt.title('Adjoint at t = 1365 seconds')
plt.xlabel('Location in ocean domain (m)')
plt.ylabel('Free surface displacement (m)')
plt.show()

plot(snapshots2[4])
plt.title('Adjoint at t = 525 seconds')
plt.xlabel('Location in ocean domain (m)')
plt.ylabel('Free surface displacement (m)')
plt.show()

plot(snapshots2[5])
plt.title('Adjoint at t = 0 seconds')
plt.xlabel('Location in ocean domain (m)')
plt.ylabel('Free surface displacement (m)')
plt.show()

if (vid == 'y'):
    plot(video2)
    plt.xlabel('Location in ocean domain (m)')
    plt.ylabel('Free surface displacement (m)')
    plt.show()
