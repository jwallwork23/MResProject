from thetis import *
import scipy.interpolate
from scipy.io.netcdf import NetCDFFile
from math import radians
import numpy as np 

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

########################### USER INPUT ################################

# Specify time parameters:
dt = float(raw_input('Specify timestep (s) (default 15):') or 15)
ndump = 4
t_export = dt*ndump
T = float(raw_input('Specify time period (s) (default 7200):') or 7200)

############################## SETUP ##################################

# Define mesh (courtesy of QMESH) and function space:
mesh = Mesh("meshes/Cartesian_Tohoku.msh")     # Japanese coastline
mesh_coords = mesh.coordinates.dat.data
P1_2d = FunctionSpace(mesh, 'CG', 1)

# Construct functions to hold initial surface elevation and bathymetry:
b = Function(P1_2d, name = 'Bathymetry')
elev_init = Function(P1_2d, name = 'Initial elevation')

# Read and interpolate initial surface data (courtesy of Saito):
nc1 = NetCDFFile('Saito_files/init_profile.nc', mmap=False)
lon1 = nc1.variables['x'][:]
lat1 = nc1.variables['y'][:]
x1, y1 = vectorlonlat2tangentxy(lon1, lat1, 143., 37.)
elev1 = nc1.variables['z'][:,:]
interpolator_surf = scipy.interpolate.RectBivariateSpline(y1, x1, elev1)
elev_init_vec = elev_init.dat.data
assert mesh_coords.shape[0]==elev_init_vec.shape[0]

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
