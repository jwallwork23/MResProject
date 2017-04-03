from firedrake import *

################################# USER INPUT ###################################

# Specify problem parameters:
dt = input('Specify timestep (0.01 recommended):')
Dt = Constant(dt)
n = input('Specify number of mesh cells per m (30 recommended):')
T = input('Specify simulation duration in s (40 recommended):')

################################### FE SETUP ###################################

# Set physical and numerical parameters for the scheme
g = 9.81            # Gravitational acceleration

# Define domain and mesh
lx = 4
ly = 1
nx = lx*n
ny = ly*n
mesh = RectangleMesh(nx, ny, lx, ly)

# Define function spaces
Vmu  = VectorFunctionSpace(mesh, "CG", 2)    # Use Taylor-Hood elements
Ve = FunctionSpace(mesh, "CG", 1)           
W = MixedFunctionSpace((Vmu, Ve))

# Construct a function to store our two variables at time n
w_ = Function(W)        # Split means we can interpolate the 
mu_, eta_ = w_.split()   # initial condition into the two components

# Interpolate bathymetry
x = SpatialCoordinate(mesh)
b = Function(Ve, name = 'Bathymetry')
b.interpolate(0.1 + 0.04 * sin(2*pi*x[0]) * sin(2*pi*x[1]))
File("../screenshots/bathymetry.pvd").write(b)

####################### INITIAL AND BOUNDARY CONDITIONS ########################

# Interpolate ICs
mu_.interpolate(Expression([0, 0]))
eta_.interpolate(-0.01*cos(0.5*pi*x[0]))

# Apply no-slip BCs on the top and bottom edges of the domain
#bc1 = DirichletBC(W.sub(0), (0.0,0.0), (3,4))

################################# WEAK PROBLEM #################################

# Build the weak form of the timestepping algorithm, expressed as a 
# mixed nonlinear problem
v, xi = TestFunctions(W)
w = Function(W)
w.assign(w_)

# Here we split up a function so it can be inserted into a UFL
# expression
mu, eta = split(w)      
mu_, eta_ = split(w_)

# Establish forms (functions of the output w1), noting we only have a linear
# equation if the stong form is written in terms of a matrix:
L = (
    (xi*(eta-eta_) - Dt*inner(mu, grad(xi)) + \
    inner(mu-mu_, v) + Dt*g*b*inner(grad(eta), v))*dx   
    )

# Set up the nonlinear problem
uprob = NonlinearVariationalProblem(L, w)
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

# The function 'split' has two forms: now use the form which splits a 
# function in order to access its data
mu_, eta_ = w_.split()
mu, eta = w.split()

################################# TIMESTEPPING #################################

# Store multiple functions:
mu.rename("Fluid momentum")
eta.rename("Free surface displacement")

# Choose a final time and initialise arrays, files and dump counter:
ufile = File('outputs/model_prob3_linear.pvd')
t = 0.0
ufile.write(mu, eta, time=t)
ndump = 10
dumpn = 0

# Enter the timeloop:
while (t < T - 0.5*dt):
    t += dt
    print "t = ", t, " seconds"
    usolver.solve()
    w_.assign(w)
    dumpn += 1          # Dump the data
    if dumpn == ndump:
        dumpn -= ndump
        ufile.write(mu, eta, time=t)
