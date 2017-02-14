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
w0 = Function(W)            # Split means we can interpolate the 
u0, e0 = w0.split()         # initial condition into the two components

### INITIAL AND BOUNDARY CONDITIONS ###

# Interpolate intial steady state
u0.interpolate(Expression([0, 0]))
e0.interpolate(Expression(0))

# Establish a BC object for the oscillating inflow condition
bcval = Constant(0.0)
bc1 = DirichletBC(W.sub(1), bcval, 1)

# Apply no-slip BC to eta on the right end of the domain
bc2 = DirichletBC(W.sub(1), (0.0), 2)

# Apply no-slip BCs on u on the top and bottom edges of the domain
#bc3 = DirichletBC(W.sub(0), (0.0,0.0), (3,4))

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

# Define the outward pointing normal to the mesh
n = FacetNormal(mesh)

# Establish the bilinear form - a function of the output function w1
Lu_int = (inner(u1-u0, v) + Dt*(inner(dot(u1, nabla_grad(u1)), v)
    + nu*inner(grad(u1), grad(v)) + g*inner(grad(e1), v))
    + Dt*Cb*sqrt(dot(u0,u0))*inner(u1/(e1+h),v))*dx(degree=4)
Le_int = (xi*(e1-e0) - Dt*inner((e1+h)*u1, grad(xi)))*dx(degree=4)
L_sides = Dt*(-inner(dot(n,nabla_grad(u1)),v)
    + dot(u1, n)*(xi*(e1+h)))*ds(degree=4) # Try ds(1) or ds(2)
L = Lu_int + Le_int + L_sides

# Set up the nonlinear problem
uprob = NonlinearVariationalProblem(L, w1, bcs=[bc1, bc2])
usolver = NonlinearVariationalSolver(uprob,
        solver_parameters={
                            'ksp_type': 'gmres',
                            'ksp_rtol': '1e-8',
                            'pc_type': 'fieldsplit',
                            'pc_fieldsplit_type': 'schur',
                            'pc_fieldsplit_schur_fact_type': 'full',
                            'fieldsplit_0_ksp_type': 'cg',
                            'fieldsplit_0_pc_type': 'lu',
                            'fieldsplit_1_ksp_type': 'cg',
                            'fieldsplit_1_pc_type': 'hypre',
                            'pc_fieldsplit_schur_precondition': 'selfp',
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
T = 40.0
ufile = File('plots/source_inflow_SW.pvd')
t = 0.0
ufile.write(u1, e1, time=t)

# Initialise a dump counter and enter the timeloop, writing to file at
# each dump
ndump = 10
dumpn = 0
while (t < T - 0.5*dt):
    t += dt
    print "t = ", t
    bcval.assign(0.01*sin(2*pi*2*t)) # Update BC
    # To implement the timestepping algorithm, call the solver and 
    # assign w1 to w0.
    usolver.solve()
    w0.assign(w1)
    # Dump the data
    dumpn += 1
    if dumpn == ndump:
        dumpn -= ndump
        ufile.write(u1, e1, time=t)
