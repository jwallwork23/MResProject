from firedrake import *
from thetis import *
import scipy.interpolate as si
from scipy.io.netcdf import NetCDFFile

########################### USER INPUT ################################

# Specify problem parameters:
mode = raw_input('Use linear or nonlinear equations? (l/n): ') or 'l'
if ((mode != 'l') & (mode != 'n')):
    raise ValueError('Please try again, choosing l or n.')

# Specify time parameters:
Ts = float(raw_input('Specify timescale (s) (default 15):') or 15)
dt = Ts             # Timestep, chosen small enough for stability (s)
Dt = Constant(dt)
ndump = 4
t_export = dt*ndump
T = float(raw_input('Specify time period (s) (default 7200):') or 7200)
# INCLUDE FUNCTIONALITY FOR MESH CHOICE

############################ FE SETUP #################################

# Establish dimensional scales:
Lx = 93453.18	            # 1 deg. longitude (m) at 33N (hor. l.s.)
Ly = 110904.44	            # 1 deg. latitude (m) at 33N (ver. l.s.)
Lm = 1/sqrt(Lx**2 + Ly**2)  # Inverse magnitude of length scales
                        # Mass scale?

# Set physical and numerical parameters for the scheme:
nu = 1e-3           # Viscosity
g = 9.81            # Gravitational acceleration
Cb = 0.0025         # Bottom friction coefficient
depth = 0.1         # Specify tank water depth

# Define mesh (courtesy of QMESH) and function spaces:
mesh = Mesh("meshes/point1_point5_point5.msh")     # Japanese coastline
mesh_coords = mesh.coordinates.dat.data
Vu = VectorFunctionSpace(mesh, "CG", 2) # \ Use Taylor-Hood elements
Ve = FunctionSpace(mesh, "CG", 1)       # /
Vq = MixedFunctionSpace((Vu, Ve))       # We have a mixed FE problem

# Construct functions to store dependent variables and bathymetry:
q_ = Function(Vq)  
u_, eta_ = q_.split()
b = Function(Vq.sub(1), name="Bathymetry")   # Bathymetry function

################## INITIAL AND BOUNDARY CONDITIONS ####################

# Read and interpolate initial surface data (courtesy of Saito):
nc1 = NetCDFFile('Saito_files/init_profile.nc', mmap=False)
lon1 = nc1.variables['x'][:]
lat1 = nc1.variables['y'][:]
elev1 = nc1.variables['z'][:,:]
interpolator_surf = si.RectBivariateSpline(lat1, lon1, elev1)
eta_vec = eta_.dat.data
assert mesh_coords.shape[0]==eta_vec.shape[0]

# Read and interpolate bathymetry data (courtesy of GEBCO):
nc2 = NetCDFFile('bathy_data/GEBCO_bathy.nc', mmap=False)
lon2 = nc2.variables['lon'][:]
lat2 = nc2.variables['lat'][:]
elev2 = nc2.variables['elevation'][:,:]
interpolator_bath = si.RectBivariateSpline(lat2, lon2, elev2)
b_vec = b.dat.data
assert mesh_coords.shape[0]==b_vec.shape[0]

# Interpolate data onto initial surface and bathymetry profiles:
for i,p in enumerate(mesh_coords):
    eta_vec[i] = interpolator_surf(p[1], p[0])
    b_vec[i] = - interpolator_surf(p[1], p[0]) - \
               interpolator_bath(p[1], p[0])

# Post-process the bathymetry to have minimum depth of 30m:
b.assign(conditional(lt(30, b), b, 30))

# Plot initial surface and bathymetry profiles:
ufile = File('plots/init_surf.pvd')
ufile.write(eta_)
ufile = File('plots/tsunami_bathy.pvd')
ufile.write(b)

########################## WEAK PROBLEM ###############################

# Build the weak form of the timestepping algorithm, expressed as a 
# mixed nonlinear problem:
v, ze = TestFunctions(Vq)
q = Function(Vq)
q.assign(q_)
u, eta = split(q)       # \ Here split means we split up a function so
u_, eta_ = split(q_)    # / it can be inserted into a UFL expression

# Establish form:

def nonlinear_form():
    L = (
     (ze * (eta-eta_) - Lm * Dt * inner((eta + b) * u, grad(ze)) + \
      Lm * inner(u-u_, v) + \
      Lm * Lm * Dt * inner(dot(u, nabla_grad(u)), v) + \
      Lm * nu * inner(grad(u), grad(v)) + \
      Lm * Lm * Dt * g * inner(grad(eta), v) + \
      Lm * Lm * Dt * Cb * sqrt(dot(u_, u_)) * inner(u/(eta+b), v)) * \
     dx(degree=4)   
     )
    return L

def linear_form():
    L = (
    (ze * (eta-eta_) - Lm * Dt * inner((eta + b) * u, grad(ze)) + \
    Lm * inner(u-u_, v) + Lm * Lm * Dt * g *(inner(grad(eta), v))) * dx
    )
    return L

if (mode == 'l'):
    L = linear_form()
elif (mode == 'n'):
    L = nonlinear_form()

# Set up the variational problem
uprob = NonlinearVariationalProblem(L, q)
usolver = NonlinearVariationalSolver(uprob,
        solver_parameters={
                            'ksp_type': 'gmres',
                            'ksp_rtol': '1e-8',
                            'pc_type': 'fieldsplit',
                            'pc_fieldsplit_type': 'schur',
                            'pc_fieldsplit_schur_fact_type': 'full',
                            'fieldsplit_0_ksp_type': 'cg',
                            'fieldsplit_0_pc_type': 'ilu',
                            'fieldsplit_1_ksp_type': 'cg',
                            'fieldsplit_1_pc_type': 'hypre',
                            'pc_fieldsplit_schur_precondition': 'selfp',
                            })

# The function 'split' has two forms: now use the form which splits a 
# function in order to access its data
u_, eta_ = q_.split()
u, eta = q.split()

############################ TIMESTEPPING #############################

# Store multiple functions
u.rename('Fluid velocity')
eta.rename('Free surface displacement')

# Initialise arrays, files and dump counter
if (mode == 'l'):
    ufile = File('tsunami_outputs/tohoku_linear.pvd')
elif (mode == 'n'):
    ufile = File('tsunami_outputs/tohoku_nonlinear.pvd')
t = 0.0
ufile.write(u, eta, time=t)
ndump = 10
dumpn = 0

# Create a dictionary containing checkpointed values of eta:
checks ={0.0: eta}

# Enter the timeloop:
while (t < T - 0.5*dt):     
    t += dt
    print 't = ', t/60, ' mins'
    ## CALCULATE log_2(eta_max) to evaluate damage at coast
    usolver.solve()
    q_.assign(q)
    dumpn += 1              # Dump the data
    if dumpn == ndump:
        dumpn -= ndump
        ufile.write(u, eta, time=t)
        # TODO: MAKE THIS MORE GENERAL
        checks[float(int(20*t))/20.0 + 0.05] = eta

print len(checks.keys())    # Sanity check

############################ THETIS SETUP #############################

# Construct solver:
solver_obj = solver2d.FlowSolver2d(mesh, b)
options = solver_obj.options
options.t_export = t_export
options.t_end = T
options.timestepper_type = 'backwardeuler'  # Use implicit timestepping
options.dt = Dt
options.outputdir = 'tsunami_outputs'

# Apply ICs:
solver_obj.assign_initial_conditions(elev=eta_)

# Run the model:
solver_obj.iterate()

# OUTPUT CHECKS FOR THETIS TOO

########################### EVALUATE ERROR ############################

##for keys in checks:
    # TO DO

