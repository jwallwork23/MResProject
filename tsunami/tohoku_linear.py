from firedrake import *
import numpy as np
import scipy.interpolate
from scipy.io.netcdf import NetCDFFile
import GFD_basisChange_tools as gfd         # Not currently used
import utm                                  # --------""--------

################################# USER INPUT ###################################

# Specify problem parameters:
Ts = input('Specify timescale (s) (default 10):') or 10
# INCLUDE FUNCTIONALITY FOR MESH CHOICE

################################# FE SETUP #####################################

# Establish dimensional scales:
Lx = 93453.18	            # 1 deg. longitude (m) at 33N (hor. length scale)
Ly = 110904.44	            # 1 deg. latitude (m) at 33N (ver. length scale)
Lm = 1/sqrt(Lx**2 + Ly**2)  # Inverse magnitude of length scales
                        # Mass scale?

# Set physical and numerical parameters for the scheme:
g = 9.81            # Gravitational acceleration (ms^{-2})
dt = Ts             # Timestep, chosen small enough for stability (s)
Dt = Constant(dt)

# Define mesh (courtesy of QMESH), function spaces and initial surface:
mesh = Mesh("meshes/point1_point5_point5.msh")     # Japanese coastline
mesh_coords = mesh.coordinates.dat.data
Vu = VectorFunctionSpace(mesh, "CG", 2)    # \ Use Taylor-Hood elements
Ve = FunctionSpace(mesh, "CG", 1)           # /
W = MixedFunctionSpace((Vu, Ve))           # We consider a mixed FE problem

# Construct functions to store dependent variables and bathymetry:
w_ = Function(W)                            # \ Here 'split' means we  
u_, eta_ = w_.split()                       # / interpolate IC into components
b = Function(W.sub(1), name="Bathymetry")   # Bathymetry function

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
ufile = File('plots/init_surf.pvd')
ufile.write(eta_)
ufile = File('plots/tsunami_bathy.pvd')
ufile.write(b)

# Interpolate IC on fluid velocity:
u_.interpolate(Expression([0, 0]))

################################# WEAK PROBLEM #################################

# Build the weak form of the timestepping algorithm, expressed as a 
# mixed linear problem:
v, xi = TestFunctions(W)
w = Function(W)
w.assign(w_)
u, eta = split(w)          # \ Here split means we split up a function so
u_, eta_ = split(w_)       # / it can be inserted into a UFL expression

# Establish the linear and bilinear forms (functions of the output w1):
L = (
    (xi * (eta-eta_) - Lm * Dt * inner((eta + b) * u, grad(xi)) + \
    Lm**2 * inner(u-u_, v) + Lm * Dt * g * (inner(grad(eta), v))) * dx
)


# Set up the nonlinear problem and specify solver parameters:
uprob = NonlinearVariationalProblem(L, w)
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
u_, eta_ = w_.split()                           # IS THIS NEEDED?
u, eta = w.split()

################################# TIMESTEPPING #################################

# Store multiple functions
u.rename("Fluid velocity")
eta.rename("Free surface displacement")

# Choose a final time and initialise arrays, files and dump counter
T = 1000.0*Ts
ufile = File('plots/simulation_linear.pvd')
t = 0.0
ufile.write(u, eta, time=t)
ndump = 5
dumpn = 0

while (t < T - 0.5*dt):     # Enter the timeloop
    t += dt
    if (t % 60 == 0):
        print "t = ", t/60, " mins"
    usolver.solve()
    w_.assign(w)
    dumpn += 1              # Dump the data
    if dumpn == ndump:
        dumpn -= ndump
        ufile.write(u, eta, time=t)
