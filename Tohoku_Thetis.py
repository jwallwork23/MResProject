from thetis import *

import numpy as np
from math import radians, sin, cos
import scipy.interpolate as si
from scipy.io.netcdf import NetCDFFile

from utm import *

def vectorlonlat2utm(lon, lat) :
    """A function which projects vectors containing longitude-latitude coordinates onto a tangent plane at (lon0, lat0) 
    in utm coordinates (x,y), with units being metres."""
    x = np.zeros((len(lon), 1))
    y = np.zeros((len(lat), 1))
    assert (len(x) == len(y))
    for i in range(len(x)) :
        x[i], y[i], zn, zl = from_latlon(lat[i], lon[i], 54)    # Force zone number 54

        # TODO: Implement a projection which doesn't restart x-coord at zone boundaries.
        # Domain x-min = 5.44e4
        # I think the x-origin of the mesh is the central 500,000m line of zone 53

        print 'Coords ', x[i], y[i], 'Zone ', zn, zl
    return x, y

# Define initial mesh (courtesy of QMESH) and functions, with initial conditions set:
res = int(raw_input('Mesh coarseness? (Integer in range 1-5): ') or 3)
if res == 1 :
    mesh = Mesh('resources/meshes/TohokuXFine.msh')
elif res == 2 :
    mesh = Mesh('resources/meshes/TohokuFine.msh')
elif res == 3 :
    mesh = Mesh('resources/meshes/TohokuMedium.msh')
elif res == 4 :
    mesh = Mesh('resources/meshes/TohokuCoarse.msh')
elif res == 5 :
    mesh = Mesh('resources/meshes/TohokuXCoarse.msh')
else : raise ValueError('Please try again, choosing an integer in the range 1-5.')
mesh_coords = mesh.coordinates.dat.data
Vu = VectorFunctionSpace(mesh, 'CG', 2)                                 # \ Use Taylor-Hood
Ve = FunctionSpace(mesh, 'CG', 1)                                       # /
Vq = MixedFunctionSpace((Vu, Ve))                                       # Mixed FE problem

# Construct functions to store inital free surface and bathymetry:
eta0 = Function(Vq.sub(1), name = 'Initial surface')
b = Function(Vq.sub(1), name = 'Bathymetry')

# Read and interpolate initial surface data (courtesy of Saito):
nc1 = NetCDFFile('resources/Saito_files/init_profile.nc', mmap = False)
lon1 = nc1.variables['x'][:]
lat1 = nc1.variables['y'][:]
elev1 = nc1.variables['z'][:, :]
x1, y1 = vectorlonlat2utm(lon1, lat1)
interpolator_surf = si.RectBivariateSpline(y1, x1, elev1)
eta0vec = eta0.dat.data
assert mesh_coords.shape[0] == eta0vec.shape[0]

# Read and interpolate bathymetry data (courtesy of GEBCO):
nc2 = NetCDFFile('resources/bathy_data/GEBCO_bathy.nc', mmap = False)
lon2 = nc2.variables['lon'][:]
lat2 = nc2.variables['lat'][:-1]
elev2 = nc2.variables['elevation'][:-1, :]
x2, y2 = vectorlonlat2utm(lon2, lat2)
interpolator_bath = si.RectBivariateSpline(y2, x2, elev2)
b_vec = b.dat.data
assert mesh_coords.shape[0] == b_vec.shape[0]

# Interpolate data onto initial surface and bathymetry profiles:
for i, p in enumerate(mesh_coords) :
    eta0vec[i] = interpolator_surf(p[1], p[0])
    b_vec[i] = - interpolator_surf(p[1], p[0]) - interpolator_bath(p[1], p[0])

# Cap bathymetry:
b.assign(conditional(lt(30, b), b, 30))

# Plot initial surface and bathymetry profiles:
File('plots/tsunami_outputs/init_surf.pvd').write(eta0)
File('plots/tsunami_outputs/tsunami_bathy.pvd').write(b)

# Specify time parameters:
dt = float(raw_input('Specify timestep (s) (default 1):') or 1)
ndump = 60
t_export = dt * ndump
T = float(raw_input('Specify time period (hours) (default 1):') or 1) * 3600.

# Construct solver:
solver_obj = solver2d.FlowSolver2d(mesh, b)
options = solver_obj.options
options.t_export = t_export
options.t_end = T
options.timestepper_type = 'ssprk33'      # 3-stage, 3rd order Strong Stability Preserving Runge Kutta timestepping
options.dt = dt
options.outputdir = 'plots/tsunami_outputs'

# Apply ICs:
solver_obj.assign_initial_conditions(elev = eta0)

# Run the model:
solver_obj.iterate()
