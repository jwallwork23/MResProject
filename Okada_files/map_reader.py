import getopt
import sys
import csv
import numpy as np
import pdb
import os
import random
from firedrake import *
from scipy.io.netcdf import NetCDFFile
import scipy.interpolate
import GFD_basisChange_tools as basischange
import Okada_b2av_Dynamic as okada

def get_surface(mesh2d, faultfilename):
    '''Establishes water surface, given latitudinal and longitudinal region
    of interest and tsunami parameters.'''

    # Compute Okada function
    X, Y, Z, Xfbar, Yfbar, Zfbar, sflength, sfwidth = \
       okada.main()
    interpolator = scipy.interpolate.RectBivariateSpline(Y, X, Z)
    P1_2d = FunctionSpace(mesh2d, 'CG', 1)
    surface2d = Function(P1_2d, name="Surface")
    xvector = mesh2d.coordinates.dat.data
    bvector = surface2d.dat.data
    assert xvector.shape[0]==bvector.shape[0]

    for i,xy in enumerate(xvector):
        bvector[i] = interpolator(xy[1]+22, xy[0]+130)

    return surface2d

def get_bathymetry(mesh2d, faultfilename):
    '''Establishes bathymetry, given latitudinal and longitudinal region
    of interest and tsunami parameters.'''

    # Read bathymetry data
    nc = NetCDFFile('../bathymetry_data/GEBCO_2014_2D_130.0_22.0_158.0_44.0.nc')
    # (Might have to set false)
    lon = nc.variables['lon'][:]
    lat = nc.variables['lat'][:]
    elev = nc.variables['elevation'][:,:]
##    nc.close()
    interpolator_bath = scipy.interpolate.RectBivariateSpline(lat, lon, elev)

    # Compute Okada function
    X, Y, Z, Xfbar, Yfbar, Zfbar, sflength, sfwidth = okada.main()
    interpolator_okada = scipy.interpolate.RectBivariateSpline(Y, X, Z)
    P1_2d = FunctionSpace(mesh2d, 'CG', 1)
    bathymetry2d = Function(P1_2d, name="Bathymetry")
    xvector = mesh2d.coordinates.dat.data
    bvector = bathymetry2d.dat.data
    assert xvector.shape[0]==bvector.shape[0]
  
    for i,xy in enumerate(xvector):
      bvector[i] = - interpolator_okada(xy[1]+22, xy[0]+130) \
                 - interpolator_bath(xy[1]+22, xy[0]+130)
  
    return bathymetry2d

## TESTING

n = 3               # Mesh resolution parameter
lat = 110904.44     # 1 degree latitude in metres in Japanese locality
lon = 93453.18      # 1 degree longitude in metres in Japanese locality

# Run file for surface
surface2d = get_surface(RectangleMesh(28*n, 22*n, 28, 22), 'fault.txt')
#surface2d = get_surface(RectangleMesh(28*n, 22*n, 28*lon, 22*lat), 'fault.txt')
ufile = File('plots/surf.pvd')
ufile.write(surface2d)

# Run file for bathymetry
##bathymetry2d = get_bathymetry(RectangleMesh(28*n, 22*n, 28, 22), 'fault.txt')
##ufile = File('plots/bath.pvd')
##ufile.write(bathymetry2d)
