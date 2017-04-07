from firedrake import *
import scipy.interpolate
from scipy.io.netcdf import NetCDFFile

# Not currently used
import GFD_basisChange_tools as gfd 
import utm  
import numpy as np  

################################# USER INPUT ###################################

# Specify time parameters:
Ts = raw_input('Specify timescale (s) (default 10):') or 10
dt = Ts             # Timestep, chosen small enough for stability (s)
Dt = Constant(dt)
ndump = 5
# INCLUDE FUNCTIONALITY FOR MESH CHOICE

################################# FE SETUP #####################################

# Establish dimensional scales:
Lx = 93453.18	            # 1 deg. longitude (m) at 33N (hor. length scale)
Ly = 110904.44	            # 1 deg. latitude (m) at 33N (ver. length scale)
Lm = 1/sqrt(Lx**2 + Ly**2)  # Inverse magnitude of length scales
                        # Mass scale?

# Set physical and numerical parameters for the scheme:
g = 9.81            # Gravitational acceleration (ms^{-2})

# Define mesh (courtesy of QMESH) and function spaces:
mesh = Mesh("meshes/point1_point5_point5.msh")     # Japanese coastline
mesh_coords = mesh.coordinates.dat.data
Vu = VectorFunctionSpace(mesh, "CG", 2)     # \ Use Taylor-Hood elements
Ve = FunctionSpace(mesh, "CG", 1)           # /
Vq = MixedFunctionSpace((Vu, Ve))           # We consider a mixed FE problem

# Construct functions to store dependent variables and bathymetry:
q_ = Function(Vq)                           # \ Here 'split' means we  
u_, eta_ = q_.split()                       # / interpolate IC into components
b = Function(Vq.sub(1), name="Bathymetry")  # Bathymetry function

############### INITIAL AND BOUNDARY CONDITIONS AND BATHYMETRY #################

# Read and interpolate initial surface data (courtesy of Saito):
nc1 = NetCDFFile('Saito_files/init_profile.nc', mmap=False)
lon1 = nc1.variables['x'][:]
lat1 = nc1.variables['y'][:]
elev1 = nc1.variables['z'][:,:]
interpolator_surf = scipy.interpolate.RectBivariateSpline(lat1, lon1, elev1)
eta_vec = eta_.dat.data
assert mesh_coords.shape[0]==eta_vec.shape[0]

# Read and interpolate bathymetry data (courtesy of GEBCO):
nc2 = NetCDFFile('bathy_data/GEBCO_bathy.nc', mmap=False)
lon2 = nc2.variables['lon'][:]
lat2 = nc2.variables['lat'][:]
elev2 = nc2.variables['elevation'][:,:]
interpolator_bath = scipy.interpolate.RectBivariateSpline(lat2, lon2, elev2)
b_vec = b.dat.data
assert mesh_coords.shape[0]==b_vec.shape[0]

# Interpolate data onto initial surface and bathymetry profiles:
for i,p in enumerate(mesh_coords):
    eta_vec[i] = interpolator_surf(p[1], p[0])
    b_vec[i] = - interpolator_surf(p[1], p[0]) - interpolator_bath(p[1], p[0])

# Post-process the bathymetry to have minimum depth of 30m:
b.assign(conditional(lt(30, b), b, 30))

# Plot initial surface and bathymetry profiles:
ufile = File('tsunami_test_outputs/init_surf.pvd')
ufile.write(eta_)
ufile = File('tsunami_test_outputs/tsunami_bathy.pvd')
ufile.write(b)

# Interpolate IC on fluid velocity:
u_.interpolate(Expression([0, 0]))

################################# WEAK PROBLEM #################################

# Build the weak form of the timestepping algorithm, expressed as a 
# mixed linear problem:
v, ze = TestFunctions(Vq)
q = Function(Vq)
q.assign(q_)
u, eta = split(q)          # \ Here split means we split up a function so
u_, eta_ = split(q_)       # / it can be inserted into a UFL expression

# Establish the linear and bilinear forms (functions of the output q):
L = (
    (ze * (eta-eta_) - Lm * Dt * inner((eta + b) * u, grad(ze)) + \
    Lm**2 * inner(u-u_, v) + Lm * Dt * g * (inner(grad(eta), v))) * dx
)


# Set up the nonlinear problem and specify solver parameters:
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

# Split dependent variables, to access data:
u_, eta_ = q_.split()                           # IS THIS NEEDED?
u, eta = q.split()

################################# TIMESTEPPING #################################

# Store multiple functions
u.rename("Fluid velocity")
eta.rename("Free surface displacement")

# Choose a final time and initialise arrays, files and dump counter
T = 1000.0*Ts
ufile = File('tsunami_test_outputs/simulation_linear.pvd')
t = 0.0
ufile.write(u, eta, time=t)
dumpn = 0

while (t < T - 0.5*dt):     # Enter the timeloop
    t += dt
    if (t % 60 == 0):
        print "t = ", t/60, " mins"
## CALCULATE log_2(eta_max) to evaluate damage at coast
    usolver.solve()
    q_.assign(q)
    dumpn += 1              # Dump the data
    if dumpn == ndump:
        dumpn -= ndump
        ufile.write(u, eta, time=t)
