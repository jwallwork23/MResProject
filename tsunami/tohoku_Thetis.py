from thetis import *
import scipy.interpolate
from scipy.io.netcdf import NetCDFFile

########################### USER INPUT ################################

# Specify time parameters:
Ts = float(raw_input('Specify timescale (s) (default 15):') or 15)
dt = Ts             # Timestep, chosen small enough for stability (s)
Dt = Constant(dt)
ndump = 4
t_export = dt*ndump
T = float(raw_input('Specify time period (s) (default 7200):') or 7200)

############################## SETUP ##################################

# Define mesh (courtesy of QMESH) and function space:
mesh = Mesh("meshes/point1_point5_point5.msh")     # Japanese coastline
mesh_coords = mesh.coordinates.dat.data
P1_2d = FunctionSpace(mesh, 'CG', 1)

# Construct functions to hold initial surface elevation and bathymetry:
b = Function(P1_2d, name = 'Bathymetry')
elev_init = Function(P1_2d, name = 'Initial elevation')

# Read and interpolate initial surface data (courtesy of Saito):
nc1 = NetCDFFile('Saito_files/init_profile.nc', mmap=False)
lon1 = nc1.variables['x'][:]
lat1 = nc1.variables['y'][:]
elev1 = nc1.variables['z'][:,:]
interpolator_surf = scipy.interpolate.RectBivariateSpline(lat1, lon1, elev1)
elev_init_vec = elev_init.dat.data
assert mesh_coords.shape[0]==elev_init_vec.shape[0]

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
    elev_init_vec[i] = interpolator_surf(p[1], p[0])
    b_vec[i] = - interpolator_surf(p[1], p[0]) - interpolator_bath(p[1], p[0])

# Construct solver:
solver_obj = solver2d.FlowSolver2d(mesh, b)
options = solver_obj.options
options.t_export = t_export
options.t_end = T
options.timestepper_type = 'backwardeuler'      # Use implicit timestepping
options.dt = dt
options.outputdir = 'tsunami_test_outputs'

# Apply ICs:
solver_obj.assign_initial_conditions(elev=elev_init)

# Run the model:
solver_obj.iterate()
