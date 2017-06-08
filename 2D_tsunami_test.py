from firedrake import *

import numpy as np
import matplotlib.pyplot as plt
from time import clock

from utils import adapt, construct_hessian, compute_steady_metric, Meshd, update_SW_FE

# Define initial (uniform) mesh:
inv_n = float(raw_input('Size of cells in km (default 10): ') or 10) * 1e3
n = 1. / inv_n
lx = 4e5
nx = int(lx * n)
mesh = SquareMesh(nx, nx, lx, lx)
meshd = Meshd(mesh)
x,y = SpatialCoordinate(mesh)
N1 = len(mesh.coordinates.dat.data)                                     # Minimum number of nodes
N2 = N1                                                                 # Maximum number of nodes
print 'Initial number of nodes : ', N1

# Simulation duration:
T = float(raw_input('Simulation duration in s (default 4200): ') or 4200.)

# Set up adaptivity parameters:
remesh = raw_input('Use adaptive meshing (y/n)?: ') or 'y'
if remesh == 'y' :
    hmin = float(raw_input('Minimum element size in km (default 5)?: ') or 5.) * 1e3
    hmax = float(raw_input('Maximum element size in km (default 100)?: ') or 100.) * 1e3
    rm = int(raw_input('Timesteps per remesh (default 10)?: ') or 10)
    nodes = float(raw_input('Target number of nodes (default 1000)?: ') or 1000.)
    ntype = raw_input('Normalisation type? (lp/manual): ') or 'lp'
    mtype = raw_input('Mesh w.r.t. speed, free surface or both? (s/f/b): ') or 'f'
    if mtype not in ('s', 'f', 'b') :
        raise ValueError('Please try again, choosing s, f or b.')
else :
    hmin = 500
    rm = int(T / dt)
    nodes = 0
    ntype = None                                                             # Timesteps per data dump
    mtype = None
    if remesh != 'n' :
        raise ValueError('Please try again, choosing y or n.')

# Courant number adjusted timestepping parameters:
ndump = 1
g = 9.81                                    # Gravitational acceleration (m s^{-2})
dt = 0.8 * hmin / np.sqrt(g * 4000.)        # Timestep length (s), using wavespeed sqrt(gh)
Dt = Constant(dt)
print 'Using Courant number adjusted timestep dt = %1.4f' % dt

# Define function spaces:
Vu = VectorFunctionSpace(mesh, 'CG', 1)                                     # TODO: consider Taylor-Hood elements
Ve = FunctionSpace(mesh, 'CG', 1)
Vq = MixedFunctionSpace((Vu, Ve))                                           # Mixed FE problem

# Establish bathymetry function:
b = Function(Ve, name = 'Bathymetry')
b.interpolate(Expression('x[0] <= 50000. ? 200. : 4000.'))  # Shelf break bathymetry

# Construct a function to store our two variables at time n:
q_ = Function(Vq)                                                           # Forward solution tuple
u_, eta_ = q_.split()

# Interpolate initial conditions, noting higher magnitude wave used due to geometric spreading:
u_.interpolate(Expression([0, 0]))
eta_.interpolate(Expression('(x[0] >= 1e5) & (x[0] <= 1.5e5) & (x[1] >= 1.8e5) & (x[1] <= 2.2e5) ? \
                                        10 * sin(pi*(x[0]-1e5) * 2e-5) * sin(pi*(x[1]-1.8e5) * 2.5e-5) : 0.'))

# Set up functions of forward weak problem:
q = Function(Vq)
q.assign(q_)
v, ze = TestFunctions(Vq)
u, eta = split(q)
u_, eta_ = split(q_)

# For timestepping we consider the implicit midpoint rule and so must create new 'mid-step' functions:
uh = 0.5 * (u + u_)
etah = 0.5 * (eta + eta_)

# Specify solver parameters:
params = {'mat_type': 'matfree',
          'snes_type': 'ksponly',
          'pc_type': 'python',
          'pc_python_type': 'firedrake.AssembledPC',
          'assembled_pc_type': 'lu',
          'snes_lag_preconditioner': -1,
          'snes_lag_preconditioner_persists': True,}

# Set up the variational problem:
L1 = (ze * (eta - eta_) - Dt * inner(b * uh, grad(ze)) +
     inner(u - u_, v) + Dt * g *(inner(grad(etah), v))) * dx
q_prob = NonlinearVariationalProblem(L1, q)
q_solv = NonlinearVariationalSolver(q_prob, solver_parameters = params)

# 'Split' functions to access their data and relabel:
u_, eta_ = q_.split()
u, eta = q.split()
u.rename('Fluid velocity')
eta.rename('Free surface displacement')

# Initialise time, counters and files:
t = 0.
dumpn = 0
mn = 0
cnt = 0
i = 0
q_file = File('plots/adjoint_test_outputs/linear_forward.pvd')
m_file = File('plots/adapt_plots/2D_tsunami_metric.pvd')
q_file.write(u, eta, time = t)
tic1 = clock()

# if remesh == 'n' :
#
#     # Establish a BC object to get 'coastline'
#     bc = DirichletBC(Vq.sub(1), 0, 1)
#     b_nodes = bc.nodes
#
#     # Initialise a CG1 version of mu and some arrays for storage:
#     V1 = VectorFunctionSpace(mesh, 'CG', 1)
#     mu_cg1 = Function(V1)
#     mu_cg1.interpolate(mu)
#     eta_vals = np.zeros((int(T / (ndump * dt)) + 1, (nx + 1) ** 2))
#     mu_vals = np.zeros((int(T / (ndump * dt)) + 1, (nx + 1) ** 2, 2))
#     m = np.zeros((int(T / (ndump * dt)) + 1))
#     eta_vals[i, :] = eta.dat.data
#     mu_vals[i, :, :] = mu_cg1.dat.data
#     m[i] = np.log2(max(max(eta_vals[i, b_nodes]), 0.5))

# Enter the forward timeloop:
while t < T - 0.5 * dt :

    # Update counters:
    mn += 1
    cnt = 0

    if remesh == 'y' :

        V = TensorFunctionSpace(mesh, 'CG', 1)

        if mtype != 'f' :
            # Establish velocity speed for adaption:
            spd = Function(FunctionSpace(mesh, 'CG', 1))
            spd.interpolate(sqrt(dot(u, u)))

            # Compute Hessian and metric:
            H = construct_hessian(mesh, V, spd)
            M = compute_steady_metric(mesh, V, H, spd, h_min = hmin, h_max = hmax, N = nodes, normalise = ntype)

        if mtype != 's' :

            # Compute Hessian and metric:
            H = construct_hessian(mesh, V, eta)
            M2 = compute_steady_metric(mesh, V, H, eta, h_min = hmin, h_max = hmax, N = nodes, normalise = ntype)

            if mtype == 'b' :
                M = metric_intersection(mesh, V, M, M2)

            else :
                M = M2

        # Adapt mesh and set up new function spaces:
        M.rename('Metric field')
        mesh_ = mesh
        meshd_ = Meshd(mesh_)
        tic2 = clock()
        mesh = adapt(mesh, M)
        meshd = Meshd(mesh)
        q_, q, u_, u, eta_, eta, b, Vq = update_SW_FE(meshd_, meshd, u_, u, eta_, eta, b)
        toc2 = clock()

        # Data analysis:
        n = len(mesh.coordinates.dat.data)
        if n < N1:
            N1 = n
        elif n > N2:
            N2 = n

        # Print to screen:
        print ''
        print '************ Adaption step %d **************' % mn
        print 'Time = %1.2fs' % t
        print 'Number of nodes after adaption step %d: ' % mn, n
        print 'Min. nodes in mesh: %d... max. nodes in mesh: %d' % (N1, N2)
        print 'Elapsed time for adaption step %d: %1.2fs' % (mn, toc2 - tic2)
        print ''

    # Set up functions of weak problem:
    v, ze = TestFunctions(Vq)
    u, eta = split(q)
    u_, eta_ = split(q_)
    uh = 0.5 * (u + u_)
    etah = 0.5 * (eta + eta_)

    # Set up the variational problem:
    L1 = (ze * (eta - eta_) - Dt * inner(b * uh, grad(ze)) +
          inner(u - u_, v) + Dt * g * (inner(grad(etah), v))) * dx
    q_prob = NonlinearVariationalProblem(L1, q)
    q_solv = NonlinearVariationalSolver(q_prob, solver_parameters = params)

    # 'Split' functions to access their data and relabel:
    u_, eta_ = q_.split()
    u, eta = q.split()
    u.rename('Fluid velocity')
    eta.rename('Free surface displacement')

        # Enter the inner timeloop:
    while cnt < rm :
        t += dt
        cnt += 1
        q_solv.solve()
        q_.assign(q)
        dumpn += 1
        if dumpn == ndump :

            dumpn -= ndump
            q_file.write(u, eta, time = t)

            if remesh == 'y' :
                m_file.write(M, time = t)
            else :
                print 't = %1.2fs, mesh number =' % t
                # i += 1
                # mu_cg1.interpolate(mu)
                # mu_vals[i, :, :] = mu_cg1.dat.data
                # eta_vals[i, :] = eta.dat.data
                #
                # # Implement damage measures:                        # TODO: change this as it is dominated by shoaling
                # m[i] = np.log2(max(max(eta_vals[i, b_nodes]), 0.5))

# End timing and print:
toc1 = clock()
if remesh == 'y':
    print 'Elapsed time for adaptive forward solver: %1.2fs' % (toc1 - tic1)
else :
    print 'Elapsed time for non-adaptive forward solver: %1.2fs' % (toc1 - tic1)

print 'Forward problem solved.... now for the adjoint problem.'

if remesh == 'y':

    # Reset uniform mesh:
    mesh = SquareMesh(nx, nx, lx, lx)

    # Re-define function spaces:
    Vu = VectorFunctionSpace(mesh, 'CG', 1)                         # TODO: Taylor-Hood elements
    Ve = FunctionSpace(mesh, 'CG', 1)                               #
    Vq = MixedFunctionSpace((Vu, Ve))                               # Mixed FE problem

    # Re-establish bathymetry function:
    b = Function(Ve, name = 'Bathymetry')
    b.interpolate(Expression('x[0] <= 50000. ? 200. : 4000.'))      # Shelf break bathymetry

# Construct a function to store our two variables at time n:
lam_ = Function(Vq)                                                 # Adjoint solution tuple
lu_, le_ = lam_.split()

# Interpolate initial and boundary conditions, noting higher magnitude wave used due to geometric spreading:
lu_.interpolate(Expression([0, 0]))
le_.interpolate(Expression('(x[0] >= 1e4) & (x[0] <= 2.5e4) & (x[1] >= 1.8e5) & (x[1] <= 2.2e5) ? 10 : 0.'))

# Set up functions of weak problem:
lam = Function(Vq)
lam.assign(lam_)
w, xi = TestFunctions(Vq)
lu, le = split(lam)
lu_, le_ = split(lam_)
luh = 0.5 * (lu + lu_)
leh = 0.5 * (le + le_)

# Set up the variational problem:
L2 = ((le - le_) * xi - Dt * g * b * inner(luh, grad(xi)) + inner(lu - lu_, w) + Dt * b * inner(grad(leh), w)) * dx
lam_prob = NonlinearVariationalProblem(L2, lam)
lam_solv = NonlinearVariationalSolver(lam_prob, solver_parameters = params)

# 'Split' functions to access their data and relabel:
lu_, le_ = lam_.split()
lu, le = lam.split()
lu.rename('Adjoint fluid velocity')
le.rename('Adjoint free surface displacement')

# Initialise counters and files:
cnt = 0
mn = 0
lam_file = File('plots/adjoint_test_outputs/linear_adjoint.pvd')
m_file2 = File('plots/adapt_plots/2D_adj_tsunami_metric.pvd')
lam_file.write(lu, le, time = 0)
tic3 = clock()

# if remesh == 'n':
#
#     # Initialise a CG1 version of lm and some arrays for storage (with dimension pre-allocated for speed):
#     lm_cg1 = Function(V1)
#     lm_cg1.interpolate(lm)
#     le_vals = np.zeros((int(T/(ndump*dt))+1, (nx+1)**2))
#     lm_vals = np.zeros((int(T/(ndump*dt))+1, (nx+1)**2, 2))
#     ql_vals = np.zeros((int(T/(ndump*dt))+1, (nx+1)**2))
#     q_dot_lam = Function(Vq.sub(1))
#     q_dot_lam.rename('Forward-adjoint inner product')
#     le_vals[i,:] = le.dat.data
#     lm_vals[i,:] = lm_cg1.dat.data
#
#     # Initialise file:
#     dot_file = File('plots/adjoint_test_outputs/inner_product.pvd')
#     dot_file.write(q_dot_lam, time = 0)
#
#     # Evaluate forward-adjoint inner products (noting mu and lm are in P2, while eta and le are in P1, so we need to
#     # evaluate at nodes):
#     ql_vals[i,:] = mu_vals[i,:,0] * lm_vals[i,:,0] + mu_vals[i,:,1] * lm_vals[i,:,1] + eta_vals[i,:] * le_vals[i,:]
#     q_dot_lam.dat.data[:] = ql_vals[i,:]
#
# Initialise dump counter:
if dumpn == 0 :
    dumpn = ndump

# Enter the backward timeloop:
while t > 0 :

    # Update counters:
    mn += 1
    cnt = 0

    if remesh == 'y' :

        V = TensorFunctionSpace(mesh, 'CG', 1)

        if mtype != 'f' :
            # Establish fluid speed for adaption:
            spd = Function(FunctionSpace(mesh, 'CG', 1))
            spd.interpolate(sqrt(dot(u, u)))

            # Compute Hessian and metric:
            H = construct_hessian(mesh, V, spd)
            M = compute_steady_metric(mesh, V, H, spd, h_min = hmin, h_max = hmax, N = nodes, normalise = ntype)

        if mtype != 's' :

            # Compute Hessian and metric:
            H = construct_hessian(mesh, V, eta)
            M2 = compute_steady_metric(mesh, V, H, eta, h_min = hmin, h_max = hmax, N = nodes, normalise = ntype)

            if mtype == 'b' :
                M = metric_intersection(mesh, V, M, M2)

            else :
                M = M2

        # Adapt mesh and update FE setup:
        mesh_ = mesh
        meshd_ = Meshd(mesh_)
        tic4 = clock()
        mesh = adapt(mesh, M)
        meshd = Meshd(mesh)
        lam_, lam, lu_, lu, le_, le, b, Vq = update_SW_FE(meshd_, meshd, lu_, lu, le_, le, b)
        toc4 = clock()

        # Data analysis:
        n = len(mesh.coordinates.dat.data)
        if n < N1:
            N1 = n
        elif n > N2:
            N2 = n

        # Print to screen:
        print ''
        print '************ Adaption step %d **************' % mn
        print 'Time = %1.2fs' % t
        print 'Number of nodes after adaption step %d: ' % mn, len(mesh.coordinates.dat.data)
        print 'Min. nodes in mesh: %d... max. nodes in mesh: %d' % (N1, N2)
        print 'Elapsed time for adaption step %d: %1.2fs' % (mn, toc2 - tic2)
        print ''

    # Set up functions of weak problem:
    w, xi = TestFunctions(Vq)
    lu, le = split(lam)
    lu_, le_ = split(lam_)
    luh = 0.5 * (lu + lu_)
    leh = 0.5 * (le + le_)

    # Establish form:
    L2 = (
        ((le - le_) * xi - Dt * g * b * inner(luh, grad(xi)) +
        inner(lu - lu_, w) + Dt * b * inner(grad(leh), w)) * dx
        )
    lam_prob = NonlinearVariationalProblem(L2, lam)
    lam_solv = NonlinearVariationalSolver(lam_prob, solver_parameters = params)

    # 'Split' functions to access their data and relabel:
    lu_, le_ = lam_.split()
    lu, le = lam.split()
    lu.rename('Adjoint fluid velocity')
    le.rename('Adjoint free surface displacement')

    # Enter the inner timeloop:
    while cnt < rm:
        t -= dt
        cnt += 1
        lam_solv.solve()
        lam_.assign(lam)
        dumpn -= 1
        if dumpn == 0 :

            dumpn += ndump
            lam_file.write(lu, le, time = T - t)                                        # Note time inversion

            if remesh == 'y':
                m_file2.write(M, time = t)
            else:
                print 't = %1.2fs, mesh number =' % t

#             if remesh == 'n':
#                 i -= 1
#                 lm_cg1.interpolate(lm)
#                 lm_vals[i, :] = lm_cg1.dat.data
#                 le_vals[i, :] = le.dat.data
#                 ql_vals[i, :] = mu_vals[i, :, 0] * lm_vals[i, :, 0] + mu_vals[i, :, 1] * lm_vals[i, :, 1] + \
#                                 eta_vals[i, :] * le_vals[i, :]
#                 q_dot_lam.dat.data[:] = ql_vals[i, :]
#                 dot_file.write(q_dot_lam, time = t)

# End timing and print:
toc3 = clock()
if remesh == 'y':
    print 'Elapsed time for adaptive adjoint solver: %1.2fs' % (toc3 - tic3)
else :
    print 'Elapsed time for non-adaptive adjoint solver: %1.2fs' % (toc3 - tic3)

                # # Plot damage measures:
# if remesh == 'n':
#     plt.rc('text', usetex=True)
#     plt.rc('font', family='serif')
#     plt.plot(range(0, int(T)+1, int(ndump*dt)), m, color='black')
#     plt.axis([0, T, -1.5, 3.5])
#     plt.axhline(-1, linestyle='--', color='blue')
#     plt.axhline(0, linestyle='--', color='green')
#     plt.axhline(1, linestyle='--', color='yellow')
#     plt.axhline(2, linestyle='--', color='orange')
#     plt.axhline(3, linestyle='--', color='red')
#     plt.annotate('Severe damage', xy=(0.7*T, 3), xytext=(0.72*T, 3.2), arrowprops=dict(facecolor='red', shrink=0.05))
#     plt.annotate('Some inland damage', xy=(0.7*T, 2), xytext=(0.72*T, 2.2),
#                  arrowprops=dict(facecolor='orange', shrink=0.05))
#     plt.annotate('Shore damage', xy=(0.7*T, 1), xytext=(0.72*T, 1.2),
#                  arrowprops=dict(facecolor='yellow', shrink=0.05))
#     plt.annotate('Very little damage', xy=(0.7*T, 0), xytext=(0.72*T, 0.2),
#                  arrowprops=dict(facecolor='green', shrink=0.05))
#     plt.annotate('No damage', xy=(0.7*T, -1), xytext=(0.72*T, -0.8), arrowprops=dict(facecolor='blue', shrink=0.05))
#     plt.xlabel(r'Time (s)')
#     plt.ylabel(r'm (dimensionless)')
#     plt.title(r'Damage measures')
#     plt.savefig('plots/tsunami_outputs/screenshots/2Ddamage_measures.png')