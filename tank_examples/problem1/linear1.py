from firedrake import *
import numpy as np

############################ FUNCTIONS ################################

# Mesh adaptivity function (courtesy of Nicolas Barral):
def adapt(mesh,metric):
    
    dim = mesh._topological_dimension
    entity_dofs = np.zeros(dim+1, dtype=np.int32)
    entity_dofs[0] = mesh.geometric_dimension()
    coordSection = mesh._plex.createSection(\
        [1], entity_dofs, perm=mesh.topology._plex_renumbering)
    
    plex = mesh._plex
    vStart, vEnd = plex.getDepthStratum(0)
    nbrVer = vEnd - vStart
##    print  "DEBUG  vStart: %d  vEnd: %d" % (vStart, vEnd)
##    coordSection.view()
    
    dmCoords = mesh.topology._plex.getCoordinateDM()
    dmCoords.setDefaultSection(coordSection)    
##    dmCoords.setDefaultSection(\
##        mesh.coordinates.function_space()._dm.getDefaultSection())

    #### TEMPORARY (?) HACK to sort the metric in the right order
    ####                (waiting for Matt Knepley fix in plexadapt)
    
    met = np.ndarray(shape=metric.dat.data.shape, \
                     dtype=metric.dat.data.dtype, order='C');
    for iVer in range(nbrVer):
        off = coordSection.getOffset(iVer+vStart)/dim
#        print "DEBUG  iVer: %d  off: %d   nbrVer: %d" \
#                                   %(iVer, off, nbrVer)
        met[iVer] = metric.dat.data[off]
    for iVer in range(nbrVer):
        metric.dat.data[iVer] = met[iVer]
#    metric.dat.data.data = met.data

    with mesh.coordinates.dat.vec_ro as coords:
        mesh.topology._plex.setCoordinatesLocal(coords)
    with metric.dat.vec_ro as vec:
        newplex = dmplex.petscAdap(mesh.topology._plex, vec)

    newmesh = Mesh(newplex)

    return newmesh

########################### PARAMETERS ################################

# Specify problem parameters:
dt = float(raw_input('Timestep (default 0.1)?: ') or 0.1)
Dt = Constant(dt)
n = float(raw_input('Number of mesh cells per m (default 5)?: ') \
          or 5)
T = float(raw_input('Simulation duration in s (default 5)?: ') or 5.0)

# Set physical and numerical parameters for the scheme:
g = 9.81        # Gravitational acceleration
depth = 0.1     # Tank water depth
ndump = 1       # Timesteps per data dump
rm = 5          # Timesteps per remesh

######################## INITIAL FE SETUP #############################

# Define domain and mesh:
lx = 4
ly = 1
nx = lx*n
ny = ly*n
mesh = RectangleMesh(nx, ny, lx, ly)
x = SpatialCoordinate(mesh)

# Define function spaces:
Vu = VectorFunctionSpace(mesh, 'CG', 1)     # \ TODO: Use Taylor-Hood 
Ve = FunctionSpace(mesh, 'CG', 1)           # / elements
Vq = MixedFunctionSpace((Vu, Ve))           # Mixed FE problem
Vm = TensorFunctionSpace(mesh, 'CG', 1)     # Function space of metric

# Construct a function to store our two variables at time n:
q_ = Function(Vq)       # \ Split means we can interpolate the 
u_, eta_ = q_.split()   # / initial condition into the two components

# Construct a metric and a (constant) bathymetry function:
M = Function(Vm)
b = Function(Ve, name = 'Bathymetry')
b.assign(depth)

################## INITIAL AND BOUNDARY CONDITIONS ####################

# Interpolate ICs:
u_.interpolate(Expression([0, 0]))
eta_.interpolate(-0.01*cos(0.5*pi*x[0]))

############################ TIMESTEPPING #############################

# Initialisation:
t = 0.0
dumpn = 0
q = Function(Vq)
q.assign(q_)

while (t < T-0.5*dt):

    ############################ FE SETUP #############################

    if (t != 0.0):

        # Adapt mesh:
        M.interpolate(Expression([[n**2, 0], [0, n**2]]))   # TODO:
        mesh2 = adapt(mesh, M)                              # Hessian
        m2coords = mesh2.coordinates.dat.data

        # Establish new function spaces:
        Vm2 = TensorFunctionSpace(mesh2, 'CG', 1)   # \ TODO: use
        Vu2 = VectorFunctionSpace(mesh2, 'CG', 1)   # / Taylor-Hood
        Ve2 = FunctionSpace(mesh2, 'CG', 1)
        Vq2 = MixedFunctionSpace((Vu2, Ve2))

        # Set up new functions:
        q_2 = Function(Vq2)
        u_2, eta_2 = q_2.split()
        q2 = Function(Vq2)
        u2, eta2 = q2.split()
        b2 = Function(Ve2)
        M2 = Function(Vm2)

        # Interpolate functions:
        eta2.dat.data[:] = eta.at(m2coords)
        u2.dat.data[:] = u.at(m2coords)
        u_2.dat.data[:] = u_.at(m2coords)
        eta_2.dat.data[:] = eta_.at(m2coords)
        b2.dat.data[:] = b.at(m2coords)

        # Relabel:
        q_ = q_2
        q = q2
        u_ = u_2
        u = u2
        eta_ = eta_2
        eta = eta2
        b = b2
        mesh = mesh2
        Vm = Vm2
        M = M2
        Vu = Vu2
        Ve = Ve2
        Vq = Vq2

    ########################## WEAK PROBLEM ###########################

    # Build the weak form of the timestepping algorithm, expressed as a 
    # mixed nonlinear problem
    v, ze = TestFunctions(Vq)
    u, eta = split(q)      
    u_, eta_ = split(q_)

    # Establish forms, noting we only have a linear equation if the
    # stong form is written in terms of a matrix:
    L = (
        (ze * (eta-eta_) - Dt * inner((eta + b) * u, grad(ze)) + \
        inner(u-u_, v) + Dt * g *(inner(grad(eta), v))) * dx
        )

    # Set up the variational problem
    uprob = NonlinearVariationalProblem(L, q)
    usolver = NonlinearVariationalSolver(uprob,
            solver_parameters={
                            'mat_type': 'matfree',
                            'snes_type': 'ksponly',
                            'pc_type': 'python',
                            'pc_python_type': 'firedrake.AssembledPC',
                            'assembled_pc_type': 'lu',
                            'snes_lag_preconditioner': -1, 
                            'snes_lag_preconditioner_persists': True,
                            })

    # The function 'split' has two forms: now use the form which splits
    # a function in order to access its data
    u_, eta_ = q_.split()
    u, eta = q.split()

    # Store multiple functions
    u.rename('Fluid velocity')
    eta.rename('Free surface displacement')

    ######################### INNER TIMESTEP ##########################

    # Set up files:
    ufile = File('prob1_test_outputs/prob1_step_{y}_adapt.pvd'\
                 .format(y=int(t/(dt*rm))))
    if (t == 0.0):
        ufile.write(u, eta, time=t)
    cnt = 0

    # Enter the timeloop:
    while (cnt < rm):     
        t += dt
        print 't = ', t, ' seconds, cnt = ', cnt
        cnt += 1
        usolver.solve()
        q_.assign(q)
        dumpn += 1
        if (dumpn == ndump):
            dumpn -= ndump
            ufile.write(u, eta, time=t)
