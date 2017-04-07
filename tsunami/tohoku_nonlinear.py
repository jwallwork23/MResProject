from firedrake import *
import scipy.interpolate
from scipy.io.netcdf import NetCDFFile
from math import radians
import numpy as np 

# Not currently used
import GFD_basisChange_tools as gfd 
import utm   

############################ USEFUL FUNCTIONS ##################################

def earth_radius(lat):
    '''A function which calculates the radius of the Earth for a given
    latitude.'''
    K = 1./298.257  # Earth flatness constant
    a = 6378136.3  # Semi-major axis of the Earth (m)
    return (1 - K * (sin(radians(lat))**2)) * a

def lonlat2tangentxy(lon, lat, lon0, lat0):
    '''A function which projects longitude-latitude coordinates onto a
    tangent plane at (lon0, lat0) in Cartesian coordinates (x,y), with
    units being metres.'''
    Re = earth_radius(lat)
    Rphi = Re * cos(radians(lat))
    x = Rphi * sin(radians(lon-lon0))
    y = Rphi * (1 - cos(radians(lon-lon0))) * sin(radians(lat0)) + \
        Re * sin(radians(lat-lat0))
    return x, y

def vectorlonlat2tangentxy(lon, lat, lon0, lat0):
    '''A function which projects vectors containing longitude-latitude
    coordinates onto a tangent plane at (lon0, lat0) in Cartesian
    coordinates (x,y), with units being metres.'''
    x = np.zeros((len(lon), 1))
    y = np.zeros((len(lat), 1))
    assert (len(x) == len(y))
    for i in range(len(x)):
        x[i], y[i] = lonlat2tangentxy(lon[i], lat[i], lon0, lat0)
    return x, y    

################################# USER INPUT ###################################

# Specify time parameters:
dt = raw_input('Specify timestep (s) (default 15):') or 15
Dt = Constant(dt)
ndump = 4
# INCLUDE FUNCTIONALITY FOR MESH CHOICE

################################# FE SETUP #####################################

# Set physical and numerical parameters for the scheme:
nu = 1e-3           # Viscosity (kgs^{-1}m^{-1})
g = 9.81            # Gravitational acceleration (ms^{-2})
Cb = 0.0025         # Bottom friction coefficient (dimensionless)

# Define mesh (courtesy of QMESH) and function spaces:
mesh = Mesh("meshes/Cartesian_Tohoku.msh")     # Japanese coastline
mesh_coords = mesh.coordinates.dat.data
Vu = VectorFunctionSpace(mesh, "CG", 2)     # \ Use Taylor-Hood elements
Ve = FunctionSpace(mesh, "CG", 1)           # /
Vq = MixedFunctionSpace((Vu, Ve))           # We consider a mixed FE problem

# Construct functions to store dependent variables and bathymetry:
q_ = Function(Vq)                            # \ Here 'split' means we  
u_, eta_ = q_.split()                       # / interpolate IC into components
b = Function(Vq.sub(1), name="Bathymetry")   # Bathymetry function

############### INITIAL AND BOUNDARY CONDITIONS AND BATHYMETRY #################

# Read and interpolate initial surface data (courtesy of Saito):
nc1 = NetCDFFile('Saito_files/init_profile.nc', mmap=False)
lon1 = nc1.variables['x'][:]
lat1 = nc1.variables['y'][:]
x1, y1 = vectorlonlat2tangentxy(lon1, lat1, 143., 37.)
elev1 = nc1.variables['z'][:,:]
interpolator_surf = scipy.interpolate.RectBivariateSpline(y1, x1, elev1)
eta_vec = eta_.dat.data
assert mesh_coords.shape[0]==eta_vec.shape[0]

# Read and interpolate bathymetry data (courtesy of GEBCO):
nc2 = NetCDFFile('bathy_data/GEBCO_bathy.nc', mmap=False)
lon2 = nc2.variables['lon'][:]
lat2 = nc2.variables['lat'][:-1]
x2, y2 = vectorlonlat2tangentxy(lon2, lat2, 143., 37.)
elev2 = nc2.variables['elevation'][:-1,:]
interpolator_bath = scipy.interpolate.RectBivariateSpline(y2, x2, elev2)
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
# mixed nonlinear problem:
v, ze = TestFunctions(Vq)
q = Function(Vq)
q.assign(q_)
u, eta = split(q)           # \ Here split means we split up a function so
u_, eta_ = split(q_)        # / it can be inserted into a UFL expression

# Establish the bilinear form, a function of the output function q.
# (NOTE: We use exact integration of degree 4 polynomials used since the 
# problem we consider is 'not very nonlinear'):
L = (
     (ze * (eta-eta_) - Dt * inner((eta + b) * u, grad(ze)) + \
     inner(u-u_, v) + Dt * (inner(dot(u, nabla_grad(u)), v) + \
     nu * inner(grad(u), grad(v)) + g * inner(grad(eta), v)) + \
     Dt * Cb * sqrt(dot(u_, u_)) * inner(u/(eta+b), v)) * dx(degree=4)   
 )

# Set up the nonlinear problem and specify solver parameters:
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

# Split dependent variables, to access data:
u_, eta_ = q_.split()                           # IS THIS NEEDED?
u, eta = q.split()

################################# TIMESTEPPING #################################

# Store multiple functions
u.rename("Fluid velocity")
eta.rename("Free surface displacement")

# Choose a final time and initialise arrays, files and dump counter
T = 7200.
ufile = File('tsunami_test_outputs/simulation.pvd')
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
