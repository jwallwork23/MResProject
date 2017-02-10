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
mesh = Mesh("hexagon.msh")

# Define function spaces
Vu  = VectorFunctionSpace(mesh, "CG", 2)    # Use Taylor-Hood elements
Ve = FunctionSpace(mesh, "CG", 1)           
W = MixedFunctionSpace((Vu, Ve))            

# Construct a function to store our two variables at time n
w0 = Function(W)            # Split means we can interpolate the 
u0, e0 = w0.split()         # initial condition into the two components

### INITIAL AND BOUNDARY CONDITIONS ###

# Interpolate ICs
u0.interpolate(Expression([0, 0]))
e0.interpolate(Expression('0.01*exp(-pow(x[0]+0.1,2)/0.01)*exp(-pow(x[1],2)/0.01)'))

### WEAK PROBLEM ###

# Build the weak form of the timestepping algorithm, expressed as a 
# mixed nonlinear problem
v, xi = TestFunctions(W)
w1 = Function(W)
w1.assign(w0)

# Here we split up a function so it can be inserted into a UFL
# expression
u1, e1 = split(w1)      
u0, e0 = split(w0)

# Establish the bilinear form - a function of the output function w1.
# We use exact integration of degree 4 polynomials used since the 
# problem we consider is not very nonlinear
L = (
    (xi*(e1-e0) - Dt*inner((e1+h)*u1, grad(xi))\
    + inner(u1-u0, v) + Dt*(inner(dot(u1, nabla_grad(u1)), v)\
    + nu*inner(grad(u1), grad(v)) + g*inner(grad(e1), v))
    + Dt*Cb*sqrt(dot(u0,u0))*inner(u1/(e1+h),v))*dx(degree=4)   
)   

# Set up the nonlinear problem
uprob = NonlinearVariationalProblem(L, w1)
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
u0, e0 = w0.split()
u1, e1 = w1.split()

### TIMESTEPPING ###

# Store multiple functions
u1.rename("Fluid velocity")
e1.rename("Free surface displacement")

# Choose a final time and initialise arrays and files
T = 10.0
ufile = File('plots/hex_SW.pvd')
t = 0.0
ufile.write(u1, e1, time=t)

# Initialise a dump counter and enter the timeloop, writing to file at
# each dump
ndump = 10
dumpn = 0
while (t < T - 0.5*dt):
    t += dt
    print "t = ", t
    # To implement the timestepping algorithm, call the solver and 
    # assign w1 to w0.  
    usolver.solve()
    w0.assign(w1)
    # Dump the data
    dumpn += 1
    if dumpn == ndump:
        dumpn -= ndump
        ufile.write(u1, e1, time=t)
