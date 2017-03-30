from firedrake import *

############################## USEFUL FUNCTIONS ################################

# For imposing time-dependent BCs:
def wave_machine(t, A, p, in_flux):
    """Time-dependent flux function"""
    return A * sin(2 * pi * t / p) + in_flux

################################### FE SETUP ###################################

# Set physical and numerical parameters for the scheme
nu = 1e-3           # Viscosity
g = 9.81            # Gravitational acceleration
Cb = 0.0025         # Bottom friction coefficient
b = 0.1             # (Flat) bathymetry of tank
dt = 0.01           # Timestep, chosen small enough for stability
Dt = Constant(dt)
T = 40.0            # End-time of simulation
A = 0.01            # 'Tide' amplitude
p = 0.5             # 'Tide' period
in_flux = 0         # Flux into domain

# Define domain and mesh
n = 30
lx = 4
ly = 1
nx = lx*n
ny = ly*n
mesh = RectangleMesh(nx, ny, lx, ly)

# Define function spaces
Vu  = VectorFunctionSpace(mesh, "CG", 2)    # Use Taylor-Hood elements
Ve = FunctionSpace(mesh, "CG", 1)           
W = MixedFunctionSpace((Vu, Ve))            

# Construct a function to store our two variables at time n
w_ = Function(W)        # Split means we can interpolate the 
u_, eta_ = w_.split()   # initial condition into the two components

####################### INITIAL AND BOUNDARY CONDITIONS ########################

# Interpolate intial steady state
u_.interpolate(Expression([0, 0]))
eta_.interpolate(Expression(0))

# Establish a BC object for the oscillating inflow condition
bcval = Constant(0.0)
bc1 = DirichletBC(W.sub(1), bcval, 1)

# Apply no-slip BC to eta on the right end of the domain
bc2 = DirichletBC(W.sub(1), (0.0), 2)

# Apply no-slip BCs on u on the top and bottom edges of the domain
#bc3 = DirichletBC(W.sub(0), (0.0,0.0), (3,4))

################################# WEAK PROBLEM #################################

# Build the weak form of the timestepping algorithm, expressed as a 
# mixed nonlinear problem
v, xi = TestFunctions(W)
w = Function(W)
w.assign(w_)

# Here we split up a function so it can be inserted into a UFL
# expression
u, eta = split(w)      
u_, eta_ = split(w_)

# Define the outward pointing normal to the mesh
n = FacetNormal(mesh)

# Integrate terms of the momentum equation over the interior:
Lu_int = (inner(u-u_, v) + Dt*(inner(dot(u, nabla_grad(u)), v)
    + nu*inner(grad(u), grad(v)) + g*inner(grad(eta), v))
    + Dt*Cb*sqrt(dot(u_, u_))*inner(u/(eta+b), v))*dx(degree=4)
# Integrate terms of the continuity equation over the interior:
Le_int = (xi*(eta-eta_) - Dt*inner((eta+b)*u, grad(xi)))*dx(degree=4)
# Integrate over left-hand boundary:
L_side1 = Dt*(-inner(dot(n, nabla_grad(u)), v)
    + dot(u, n)*(xi*(eta+b)))*ds(1)(degree=4)
# Integrate over right-hand boundary:
L_side2 = Dt*(-inner(dot(n, nabla_grad(u)), v)
    + dot(u, n)*(xi*(eta+b)))*ds(2)(degree=4)
# Establish the bilinear form using the above integrals:
L = Lu_int + Le_int + L_side1 + L_side2

# Set up the nonlinear problem
uprob = NonlinearVariationalProblem(L, w, bcs=[bc1, bc2])
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
u_, eta_ = w_.split()
u, eta = w.split()

################################# TIMESTEPPING #################################

# Store multiple functions:
u.rename("Fluid velocity")
eta.rename("Free surface displacement")

# Choose a final time and initialise arrays, files and dump counter:
ufile = File('outputs/model_prob2.pvd')
t = 0.0
ufile.write(u, eta, time=t)
ndump = 10
dumpn = 0

# Enter the timeloop:
while (t < T - 0.5*dt):
    t += dt
    print "t = ", t, " seconds"
    bcval.assign(wave_machine(t, A, p, in_flux))    # Update BC
    usolver.solve()
    w_.assign(w)
    dumpn += 1                                      # Dump the data
    if dumpn == ndump:
        dumpn -= ndump
        ufile.write(u, eta, time=t)
