from firedrake import *

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
import scipy.interpolate as si
from scipy.io.netcdf import NetCDFFile
from math import radians
from time import clock

from utils import *

mesh_converter('resources/meshes/LonLatTohokuCoarse.msh', 143., 37.)
mesh = Mesh('resources/meshes/CartesianTohoku.msh')
mesh_coords = mesh.coordinates.dat.data
Vu = FunctionSpace(mesh, 'CG', 2)
Vv = FunctionSpace(mesh, 'CG', 2)
Ve = FunctionSpace(mesh, 'CG', 1)
Vq = MixedFunctionSpace((Vu, Vv, Ve))

# Construct functions to store forward and adjoint variables, along with bathymetry:
q_ = Function(Vq)
u_, v_, eta_ = q_.split()
eta0 = Function(Vq.sub(2), name = 'Initial surface')
b = Function(Vq.sub(2), name = 'Bathymetry')

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
b.assign(conditional(lt(30, b), b, 30))

# Simulation duration:
T = float(raw_input('Simulation duration in hours (default 1)?: ') or 1.) * 3600.
ndump = 4
dt = 15
Dt = Constant(dt)

# Set pressure gauge locations and arrays:
P02x, P02y = lonlat2tangentxy(142.5, 38.5, 143, 37)
P06x, P06y = lonlat2tangentxy(142.6, 38.7, 143, 37)
gauge = raw_input('Gauge P02 or P06?: ') or 'P02'
if gauge == 'P02' :
    gcoord = [P02x, P02y]
elif gauge == 'P06' :
    gcoord = [P06x, P06y]
else :
    raise ValueError('Gauge not recognised. Please choose P02 or P06.')

# Set parameters:
Om = 7.291e-5
f = 2 * Om * sin(radians(37))
nu = 1e-3    # Viscosity (kg s^{-1} m^{-1})
Cb = 0.0025  # Bottom friction coefficient (dimensionless)

# Specify solver parameters:
params = {'mat_type': 'matfree',
          'snes_type': 'ksponly',
          'pc_type': 'python',
          'pc_python_type': 'firedrake.AssembledPC',
          'assembled_pc_type': 'lu',
          'snes_lag_preconditioner': -1,
          'snes_lag_preconditioner_persists': True,}

dict = {0 : 'Linear, non-rotational',
        1 : 'Linear, rotational',
        2 : 'Nonlinear, nonrotational',
        3 : 'Nonlinear, rotational'}

timings = []

for key in dict :

    print ''
    print '****************', dict[key], ' case ****************'
    print ''

    # Assign initial surface and post-process the bathymetry to have a minimum depth of 30m:
    u_.interpolate(Expression(0))
    v_.interpolate(Expression(0))
    eta_.assign(eta0)

    # Set up functions of the weak problem:
    q = Function(Vq)
    q.assign(q_)
    w, z, ze = TestFunctions(Vq)
    u, v, eta = split(q)
    u_, v_, eta_ = split(q_)

    L = (ze * (eta - eta_) - Dt * b * (u * ze.dx(0) + v * ze.dx(1))
         + (u - u_) * w + Dt * g * eta.dx(0) * w
         + (v - v_) * z + Dt * g * eta.dx(1) * z) * dx
    if key in (1, 3) :                      # Rotational cases
        L += Dt * f * (u * z - v * w) * dx
    if key in (2, 3) :                    # Nonlinear cases
        L += Dt * (- eta * (u * ze.dx(0) + v * ze.dx(1))
                   + (u * u.dx(0) + v * u.dx(1)) * w + (u * v.dx(0) + v * v.dx(1)) * z
                   + nu * (u.dx(0) * w.dx(0) + u.dx(1) * w.dx(1) + v.dx(0) * z.dx(0) + v.dx(1) * z.dx(1))
                   + Cb * sqrt(u_ * u_ + v_ * v_) * (u * w + v * z) / (eta + b)) * dx
    q_prob = NonlinearVariationalProblem(L, q)
    q_solv = NonlinearVariationalSolver(q_prob, solver_parameters = params)

    # 'Split' functions in order to access their data and then relabel:
    u_, v_, eta_ = q_.split()
    u, v, eta = q.split()
    u.rename('Fluid x-velocity')
    v.rename('Fluid y-velocity')
    eta.rename('Free surface displacement')

    # Initialise counters and files:
    t = 0.
    dumpn = 0
    q_file = File('plots/tsunami_outputs/model_verif_{y}.pvd'.format(y = key))
    q_file.write(u, v, eta, time = t)
    gauge_dat = [eta.at(gcoord)]
    tic1 = clock()

    while t < T - 0.5 * dt :
        t += dt
        print 't = %1.1fs' % t
        q_solv.solve()
        q_.assign(q)
        dumpn += 1
        gauge_dat.append(eta.at(gcoord))
        if dumpn == ndump :
            dumpn -= ndump
            q_file.write(u, v, eta, time = t)

    # End timing and print:
    toc1 = clock()
    timings.append(toc1 - tic1)

    # Plot pressure gauge time series:
    plt.rc('text', usetex = True)
    plt.rc('font', family = 'serif')
    plt.plot(np.linspace(0, 60, len(gauge_dat)), gauge_dat, label = dict[key])
    plt.gcf().subplots_adjust(bottom = 0.15)
    plt.legend()
    plt.xlabel(r'Time elapsed (mins)')
    plt.ylabel(r'Free surface (m)')

print ''
for key in dict :
    print dict[key], 'case time =  %1.1fs' % timings[key]

plt.savefig('plots/tsunami_outputs/screenshots/gauge_timeseries_{y}.png'.format(y = gauge))