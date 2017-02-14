from firedrake import *

### FE SETUP ###

# Set physical and numerical parameters for the scheme
nu = 1e-3           # Viscosity
g = 9.81            # Gravitational acceleration
Cb = 0.0025         # Bottom friction coefficient
h = 0.1             # Mean water depth of tank
dt = 0.01           # Timestep, chosen small enough for stability                          
Dt = Constant(dt)

# Define domain and mesh
n = 30
mesh = RectangleMesh(4*n, n, 4, 1)

# Define function spaces
Vu  = VectorFunctionSpace(mesh, "CG", 2)    # Use Taylor-Hood elements
Ve = FunctionSpace(mesh, "CG", 1)           
W = MixedFunctionSpace((Vu, Ve))            

# Construct a function to store our two variables at time n
w_ = Function(W)            # Split means we can interpolate the 
u_, eta_ = w_.split()         # initial condition into the two components

### INITIAL AND BOUNDARY CONDITIONS ###

# Interpolate ICs
u_.interpolate(Expression([0, 0]))
eta_.interpolate(Expression('0.01*sin(0.5*pi*x[0])'))

# Apply no-slip BCs on the top and bottom edges of the domain
#bc1 = DirichletBC(W.sub(0), (0.0,0.0), (3,4))

### WEAK PROBLEM ###

# Build the weak form of the timestepping algorithm, expressed as a 
# mixed nonlinear problem
v, xi = TestFunctions(W)
w = Function(W)
w.assign(w_)

# Here we split up a function so it can be inserted into a UFL
# expression
u, eta = split(w)      
u_, eta_ = split(w_)

# Establish the bilinear form - a function of the output function w1.
# We use exact integration of degree 4 polynomials used since the 
# problem we consider is not very nonlinear
L = (
    (xi*(eta-eta_) - Dt*inner((eta+h)*u, grad(xi))\
    + inner(u-u_, v) + Dt*(inner(dot(u, nabla_grad(u)), v)\
    + nu*inner(grad(u), grad(v)) + g*inner(grad(eta), v))\
    + Dt*Cb*sqrt(dot(u_, u_))*inner(u/(eta+h), v))*dx(degree=4)   
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
u_, eta_ = w_.split()
u, eta = w.split()

### TIMESTEPPING ###

# Store multiple functions
u.rename("Fluid velocity")
eta.rename("Free surface displacement")

# Choose a final time and initialise arrays, files and dump counter
T = 40.0
ufile = File('plots/tank_SW.pvd')
t = 0.0
ufile.write(u, eta, time=t)
ndump = 10
dumpn = 0

while (t < T - 0.5*dt):     # Enter the timeloop
    t += dt
    print "t = ", t
    usolver.solve()
    w_.assign(w)
    dumpn += 1              # Dump the data
    if dumpn == ndump:
        dumpn -= ndump
        ufile.write(u, eta, time=t)
