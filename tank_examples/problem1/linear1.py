from firedrake import *

########################### PARAMETERS ################################

# Specify problem parameters:
dt = float(raw_input('Specify timestep (default 0.01): ') or 0.01)
Dt = Constant(dt)
n = float(raw_input('Specify number of mesh cells per m (default 10): ') or 10)
T = float(raw_input('Specify simulation duration in s (default 5): ') or 5.0)

# Set physical and numerical parameters for the scheme:
g = 9.81            # Gravitational acceleration
depth = 0.1         # Specify tank water depth

############################ FE SETUP #################################

# Define domain and mesh:
lx = 4
ly = 1
nx = lx*n
ny = ly*n
mesh = RectangleMesh(nx, ny, lx, ly)
x = SpatialCoordinate(mesh)

# Define function spaces:
Vu  = VectorFunctionSpace(mesh, 'CG', 2)    # Use Taylor-Hood elements
Ve = FunctionSpace(mesh, 'CG', 1)           
Vq = MixedFunctionSpace((Vu, Ve))            

# Construct a function to store our two variables at time n:
q_ = Function(Vq)       # \ Split means we can interpolate the 
u_, eta_ = q_.split()   # / initial condition into the two components

# Construct a (constant) bathymetry function:
b = Function(Ve, name = 'Bathymetry')
b.assign(depth)

################## INITIAL AND BOUNDARY CONDITIONS ####################

# Interpolate ICs:
u_.interpolate(Expression([0, 0]))
eta_.interpolate(-0.01*cos(0.5*pi*x[0]))

############################ TIMESTEPPING #############################

while (t < T-0.5*dt):

        ###################### WEAK PROBLEM ###########################

    # Build the weak form of the timestepping algorithm, expressed as a 
    # mixed nonlinear problem
    v, ze = TestFunctions(Vq)
    q = Function(Vq)
    q.assign(q_)
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
                    # only rebuild the preconditioner every 10 (-1) solves:
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

        ####################### INNER TIMESTEP #########################

    # Initialise arrays, files and dump counter
    ufile = File('prob1_test_outputs/model_prob1.pvd')
    t = 0.0
    ufile.write(u, eta, time=t)
    dumpn = 0
    cnt = 0

    # Enter the timeloop:
    while (cnt < 5):     
        t += dt
        print 't = ', t, ' seconds, cnt = ', cnt
        cnt += 1
        usolver.solve()
        w_.assign(w)
        dumpn += 1
        if dumpn == ndump:
            dumpn -= ndump
            ufile.write(u, eta, time=t)
