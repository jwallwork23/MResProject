from firedrake import *

import numpy as np
import scipy.interpolate as si
from scipy.io.netcdf import NetCDFFile

from utils import *

mesh_converter('resources/meshes/LonLatTohokuCoarse.msh', 143., 37.)
mesh = Mesh('resources/meshes/CartesianTohoku.msh')
mesh_coords = mesh.coordinates.dat.data
Vu = FunctionSpace(mesh, 'CG', 2)
Vv = FunctionSpace(mesh, 'CG', 2)
Ve = FunctionSpace(mesh, 'CG', 1)
Vq = MixedFunctionSpace((Vu, Vv, Ve))                               # Mixed FE problem

# Construct functions to store forward and adjoint variables, along with bathymetry:
q_ = Function(Vq)
u_, v_, eta_ = q_.split()
eta0 = Function(Vq.sub(1), name = 'Initial surface')
b = Function(Vq.sub(1), name = 'Bathymetry')

# Read and interpolate initial surface data (courtesy of Saito):
nc1 = NetCDFFile('resources/Saito_files/init_profile.nc', mmap = False)
lon1 = nc1.variables['x'][:]
lat1 = nc1.variables['y'][:]
x1, y1 = vectorlonlat2tangentxy(lon1, lat1, 143., 37.)
elev1 = nc1.variables['z'][:, :]
interpolator_surf = si.RectBivariateSpline(y1, x1, elev1)
eta0vec = eta0.dat.data
assert mesh_coords.shape[0] == eta0vec.shape[0]

# Read and interpolate bathymetry data (courtesy of GEBCO):
nc2 = NetCDFFile('resources/bathy_data/GEBCO_bathy.nc', mmap = False)
lon2 = nc2.variables['lon'][:]
lat2 = nc2.variables['lat'][:-1]
x2, y2 = vectorlonlat2tangentxy(lon2, lat2, 143., 37.)
elev2 = nc2.variables['elevation'][:-1, :]
interpolator_bath = si.RectBivariateSpline(y2, x2, elev2)
b_vec = b.dat.data
assert mesh_coords.shape[0] == b_vec.shape[0]

# Interpolate data onto initial surface and bathymetry profiles:
for i, p in enumerate(mesh_coords) :
    eta0vec[i] = interpolator_surf(p[1], p[0])
    b_vec[i] = - interpolator_surf(p[1], p[0]) - interpolator_bath(p[1], p[0])

# Assign initial surface and post-process the bathymetry to have a minimum depth of 30m:
u_.interpolate(Expression(0))
v_.interpolate(Expression(0))
eta_.assign(eta0)
b.assign(conditional(lt(30, b), b, 30))


