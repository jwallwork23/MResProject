from firedrake import *
import pyop2 as op2             # Not currently used

import sys, os, os.path         # Not currently used
import numpy as np
from numpy import linalg as LA

######################################################### FUNCTIONS ###########################################################

def adapt(mesh, metric):
    '''A function which generates a new mesh, provided with a previous mesh and an adaptivity metric. Courtesy of Nicolas
    Barral.'''
    
    dim = mesh._topological_dimension
    entity_dofs = np.zeros(dim+1, dtype=np.int32)
    entity_dofs[0] = mesh.geometric_dimension()
    coordSection = mesh._plex.createSection([1], entity_dofs, perm=mesh.topology._plex_renumbering)
    
    plex = mesh._plex
    vStart, vEnd = plex.getDepthStratum(0)
    nbrVer = vEnd - vStart
    
    dmCoords = mesh.topology._plex.getCoordinateDM()
    dmCoords.setDefaultSection(coordSection)    
    
    met = np.ndarray(shape=metric.dat.data.shape, dtype=metric.dat.data.dtype, order='C');
    for iVer in range(nbrVer):
        off = coordSection.getOffset(iVer+vStart)/dim

        met[iVer] = metric.dat.data[off]
    for iVer in range(nbrVer):
        metric.dat.data[iVer] = met[iVer]

    with mesh.coordinates.dat.vec_ro as coords:
        mesh.topology._plex.setCoordinatesLocal(coords)
    with metric.dat.vec_ro as vec:
        newplex = dmplex.petscAdap(mesh.topology._plex, vec)

    newmesh = Mesh(newplex)

    return newmesh


def construct_hessian(mesh, sol):
    '''A function which computes the hessian of a scalar solution field with respect to the current mesh.'''

    # Construct functions:
    H = Function(Vm)            # Hessian-to-be
    sigma = TestFunction(Vm)
    nhat = FacetNormal(mesh)    # Normal vector

    # Establish and solve a variational problem associated with the Monge-Ampere equation:
    Lh = (
            inner(H, sigma) * dx + inner(div(sigma), grad(sol)) * dx - \
            (sigma[0,1] * nhat[1] * sol.dx(0) + sigma[1,0] * nhat[0] * sol.dx(1)) * ds
        )
    H_prob = NonlinearVariationalProblem(Lh, H)
    H_solv = NonlinearVariationalSolver(H_prob, solver_parameters={'snes_rtol': 1e8,
                                                                   'ksp_rtol': 1e-5,
                                                                   'ksp_gmres_restart': 20,
                                                                   'pc_type': 'sor',
                                                                   'snes_monitor': True,
                                                                   'snes_view': False,
                                                                   'ksp_monitor_true_residual': False,
                                                                   'snes_converged_reason': True,
                                                                   'ksp_converged_reason': True,})
    H_solv.solve()

    return H


def compute_steady_metric(mesh, H, sol, h_min = 0.005, h_max = 0.1, a = 100):
    '''A function which computes the steady metric for remeshing, provided with the current mesh, hessian and free surface.
    Here h_min and h_max denote the respective minimum and maxiumum tolerated side-lengths, while a denotes the maximum
    tolerated aspect ratio. This code is based on Nicolas Barral's function ``computeSteadyMetric``, from ``adapt.py``.'''

    sol_min = 0.01
    ia = 1./(a**2)
    ihmin2 = 1./(h_min**2)
    ihmax2 = 1./(h_max**2)
    M = H
    
    for i in range(mesh.topology.num_vertices()):
        
        # Generate local Hessian, scaling to avoid roundoff error and editing values:
        H_loc = H.dat.data[i] * 1/max(abs(sol.dat.data[i]), sol_min) * n2
## TODO:                         what is this mysterious scale factor? ^^
        mean_diag = 0.5 * (H_loc[0][1] + H_loc[1][0])
        H_loc[0][1] = mean_diag; H_loc[1][0] = mean_diag

        # Find eigenpairs and truncate eigenvalues:
        lam, v = LA.eig(H_loc)
        v1, v2 = v[0], v[1]
        lam1 = min(ihmin2, max(ihmax2, abs(lam[0])))
        lam2 = min(ihmin2, max(ihmax2, abs(lam[1])))
        lam_max = max(lam1, lam2)
        lam1 = max(lam1, ia * lam_max)
        lam2 = max(lam2, ia * lam_max)

        # Reconstruct edited Hessian:
        M.dat.data[i][0,0] = lam1 * v1[0] * v1[0] + lam2 * v2[0] * v2[0]
        M.dat.data[i][0,1] = lam1 * v1[0] * v1[1] + lam2 * v2[0] * v2[1]
        M.dat.data[i][1,0] = M.dat.data[i][0,1]
        M.dat.data[i][1,1] = lam1 * v1[1] * v1[1] + lam2 * v2[1] * v2[1]

    return M


def update_FE(mesh, u_, u, eta_, eta, b):
    '''A function which updates solution fields and bathymetry from one mesh to another.'''
    
    # Establish function spaces on the new mesh:
    Vm = TensorFunctionSpace(mesh, 'CG', 1)
    Vu = VectorFunctionSpace(mesh, 'CG', 1)
    Ve = FunctionSpace(mesh, 'CG', 1)
    Vq = MixedFunctionSpace((Vu, Ve))

    # Establish functions in the new spaces:
    q_2 = Function(Vq); u_2, eta_2 = q_2.split()
    q2 = Function(Vq); u2, eta2 = q2.split()
    b2 = Function(Ve)

    # Interpolate across from the previous mesh:
    u_2.dat.data[:] = u_.at(coords)
    u2.dat.data[:] = u.at(coords)
    eta_2.dat.data[:] = eta_.at(coords)
    eta2.dat.data[:] = eta.at(coords)
    b2.dat.data[:] = b.at(coords)

    return q_2, q2, u_2, u2, eta_2, eta2, b2, Vq
    

def forward_linear_solver(q_, q, u_, eta_, b, Vq):
    '''A function which solves the forward linear shallow water equations.'''

    # Build the weak form of the timestepping algorithm, expressed as a 
    # mixed nonlinear problem:
    v, ze = TestFunctions(Vq)
    u, eta = split(q)      
    u_, eta_ = split(q_)

    # Establish forms, noting we only have a linear equation if the
    # stong form is written in terms of a matrix:
    L = (ze * (eta-eta_) - Dt * inner((eta + b) * u, grad(ze)) + inner(u-u_, v) + Dt * g *(inner(grad(eta), v))) * dx

    # Set up the variational problem
    q_prob = NonlinearVariationalProblem(L, q)
    q_solve = NonlinearVariationalSolver(q_prob, solver_parameters={'mat_type': 'matfree',
                                                                    'snes_type': 'ksponly',
                                                                    'pc_type': 'python',
                                                                    'pc_python_type': 'firedrake.AssembledPC',
                                                                    'assembled_pc_type': 'lu',
                                                                    'snes_lag_preconditioner': -1,
                                                                    'snes_lag_preconditioner_persists': True,})

    # The function 'split' has two forms: now use the form which splits
    # a function in order to access its data
    u_, eta_ = q_.split(); u, eta = q.split()

    # Store multiple functions
    u.rename('Fluid velocity'); eta.rename('Free surface displacement')

    return q_, q, u_, u, eta_, eta, q_solve

######################################################## PARAMETERS ###########################################################

# Specify problem parameters:
dt = float(raw_input('Timestep (default 0.1)?: ') or 0.1)
Dt = Constant(dt)
n = int(raw_input('Number of mesh cells per m (default 16)?: ') or 16)
n2 = n**2
T = float(raw_input('Simulation duration in s (default 5)?: ') or 5.0)
remesh = raw_input('Use adaptive meshing (y/n)?: ') or 'y'
if ((remesh != 'y') & (remesh != 'n')):
    raise ValueError('Please try again, typing y or n.')

# Set physical and numerical parameters for the scheme:
g = 9.81        # Gravitational acceleration (m s^{-2})
depth = 0.1     # Tank water depth (m)
ndump = 1       # Timesteps per data dump
rm = 6          # Timesteps per remesh

######################################## INITIAL FE SETUP AND BOUNDARY CONDITIONS #############################################

# Define domain and mesh:
lx = 4; ly = 1; nx = lx*n; ny = ly*n
mesh = RectangleMesh(nx, ny, lx, ly)
x = SpatialCoordinate(mesh)

# Define function spaces:
Vu = VectorFunctionSpace(mesh, 'CG', 1)     # \ TODO: Use Taylor-Hood elements
Ve = FunctionSpace(mesh, 'CG', 1)           # / 
Vq = MixedFunctionSpace((Vu, Ve))           # Mixed FE problem

# Construct a function to store our two variables at time n:
q_ = Function(Vq)       # \ Split means we can interpolate the initial condition onto the two components 
u_, eta_ = q_.split()   # / 

# Construct a (constant) bathymetry function:
b = Function(Ve, name = 'Bathymetry')
b.assign(depth)

# Interpolate ICs:
u_.interpolate(Expression([0, 0]))
eta_.interpolate(-0.01*cos(0.5*pi*x[0]))

######################################################## TIMESTEPPING #########################################################

# Initialisation:
t = 0.0; dumpn = 0; mn = 0
q = Function(Vq)
q.assign(q_)

while (t < T-0.5*dt):

    mn += 1

    if (t != 0.0):

        # Set up metric:
        Vm = TensorFunctionSpace(mesh, 'CG', 1)
        M = Function(Vm)

        # Build Hessian and (hence) metric:
        H = construct_hessian(mesh, eta)
        if (remesh == 'y'):
            M = compute_steady_metric(mesh, H, eta)
        else:
            M.interpolate(Expression([[n2, 0], [0, n2]]))

        # Adapt mesh and update FE setup:
        mesh = adapt(mesh, M)
        coords = mesh.coordinates.dat.data
        q_, q, u_, u, eta_, eta, b, Vq = update_FE(mesh, u_, u, eta_, eta, b)

    # Solve weak problem:
    q_, q, u_, u, eta_, eta, q_solve = forward_linear_solver(q_, q, u_, eta_, b, Vq)

    # Set up files:
    q_file = File('prob1_test_outputs/prob1_step_{y}_adapt.pvd'.format(y=mn))
    if (t == 0.0):
        q_file.write(u, eta, time=t)
    cnt = 0

    # Enter the inner timeloop:
    while (cnt < rm):     
        t += dt
        print 't = ', t, ' seconds, mesh number = ', mn
        cnt += 1
        q_solve.solve()
        q_.assign(q)
        dumpn += 1
        if (dumpn == ndump):
            dumpn -= ndump
            q_file.write(u, eta, time=t)
