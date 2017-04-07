from firedrake import *

################################# USER INPUT ###################################

# Specify problem parameters:
dt = raw_input('Specify timestep (default 0.01): ') or 0.01
Dt = Constant(dt)
n = raw_input('Specify number of mesh cells per m (default 30): ') or 30
T = raw_input('Specify simulation duration in s (default 40): ') or 40

############################## USEFUL FUNCTIONS ################################

# For imposing time-dependent BCs:
def wave_machine(t, A, p, in_flux):
    '''Time-dependent flux function'''
    return A * sin(2 * pi * t / p) + in_flux

################################### FE SETUP ###################################

# Set physical and numerical parameters for the scheme:
g = 9.81            # Gravitational acceleration
depth = 0.1         # Specify tank water depth
A = 0.01            # 'Tide' amplitude
p = 0.5             # 'Tide' period
in_flux = 0         # Flux into domain

# Define domain and mesh:
lx = 4
ly = 1
nx = lx*n
ny = ly*n
mesh = RectangleMesh(nx, ny, lx, ly)

# Define function spaces:
Vu  = VectorFunctionSpace(mesh, 'CG', 2)    # Use Taylor-Hood elements
Ve = FunctionSpace(mesh, 'CG', 1)           
Vq = MixedFunctionSpace((Vu, Ve))            

# Construct a function to store our two variables at time n:
q_ = Function(Vq)        # Split means we can interpolate the 
u_, eta_ = q_.split()   # initial condition into the two components

# Construct a (constant) bathymetry function:
b = Function(Ve, name = 'Bathymetry')
b.assign(depth)

####################### INITIAL AND BOUNDARY CONDITIONS ########################

# Interpolate intial steady state
u_.interpolate(Expression([0, 0]))
eta_.interpolate(Expression(0))

# Establish a BC object for the oscillating inflow condition
bcval = Constant(0.0)
bc1 = DirichletBC(Vq.sub(1), bcval, 1)

# Apply no-slip BC to eta on the right end of the domain
bc2 = DirichletBC(Vq.sub(1), (0.0), 2)

# Apply no-slip BCs on u on the top and bottom edges of the domain
#bc3 = DirichletBC(Vq.sub(0), (0.0,0.0), (3,4))

################################# WEAK PROBLEM #################################

# Build the weak form of the timestepping algorithm, expressed as a 
# mixed nonlinear problem
v, ze = TestFunctions(Vq)
q = Function(Vq)
q.assign(q_)

# Here we split up a function so it can be inserted into a UFL
# expression
u, eta = split(q)      
u_, eta_ = split(q_)

# Define the outward pointing normal to the mesh
n = FacetNormal(mesh)

# Integrate terms of the momentum equation over the interior:
Lu_int = (inner(u-u_, v) + Dt * g *(inner(grad(eta), v))) * dx
# Integrate terms of the continuity equation over the interior:
Le_int = (ze * (eta-eta_) - Dt * inner((eta + b) * u, grad(ze))) * dx
# Integrate over left-hand boundary:
L_side1 = dot(u, n) * (ze * (eta + b)) * ds(1)
# Integrate over right-hand boundary:
L_side2 = dot(u, n) * (ze * (eta + b)) * ds(2)
# Establish the bilinear form using the above integrals:
L = Lu_int + Le_int + L_side1 + L_side2

# Set up the nonlinear problem
uprob = NonlinearVariationalProblem(L, q, bcs=[bc1, bc2])
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
u_, eta_ = q_.split()
u, eta = q.split()

################################# TIMESTEPPING #################################

# Store multiple functions:
u.rename('Fluid velocity')
eta.rename('Free surface displacement')

# Choose a final time and initialise arrays, files and dump counter:
ufile = File('prob2_test_outputs/model_prob2_linear.pvd')
t = 0.0
ufile.write(u, eta, time=t)
ndump = 10
dumpn = 0

# Enter the timeloop:
while (t < T - 0.5*dt):
    t += dt
    print 't = ', t, ' seconds'
    bcval.assign(wave_machine(t, A, p, in_flux))    # Update BC
    usolver.solve()
    q_.assign(q)
    dumpn += 1                                      # Dump the data
    if dumpn == ndump:
        dumpn -= ndump
        ufile.write(u, eta, time=t)
