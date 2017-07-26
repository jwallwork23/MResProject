from firedrake import *
from math import radians
import numpy as np
import scipy.interpolate as si
from scipy.io.netcdf import NetCDFFile
from time import clock

from utils.conversion import from_latlon, get_latitude
from utils.domain import Tohoku_domain

# Change backend to resolve framework problems:
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

print ''
print '******************************** MODEL VERIFICATION ********************************'
print ''
print 'Options...'

# Establish cases:
mode = {0: 'Linear, non-rotational', 1: 'Linear, rotational', 2: 'Nonlinear, nonrotational', 3: 'Nonlinear, rotational'}
for key in mode:
    print key, ' : ', mode[key]
print ''
choices = int(raw_input('Choose mode (0/1/2/3 or 4 to try all): ') or 4)
if choices in (0, 1, 2, 3):
    mode = {choices: mode[choices]}
coarseness = int(raw_input('Mesh coarseness? (Integer in range 1-5): ') or 3)

# Define mesh, mixed function space and variables:
mesh, W, q_, u_, v_, eta_, lam_, lu_, lv_, b = Tohoku_domain(res=coarseness, split='y')
eta0 = Function(W.sub(2), name='Initial free surface')
eta0.assign(eta_)
coords = mesh.coordinates.dat.data
print '........ mesh loaded'

# Set physical parameters:
Om = 7.291e-5                   # Rotation rate of Earth (rad s^{-1})
g = 9.81                        # Gravitational acceleration (m s^{-2})
nu = 1e-3                       # Viscosity (kg s^{-1} m^{-1})
Cb = 0.0025                     # Bottom friction coefficient (dimensionless)

# Simulation duration:
T = float(raw_input('Simulation duration in hours (default 1)?: ') or 1.) * 3600.
dt = float(raw_input('Specify timestep in seconds (default 1): ') or 1.)
Dt = Constant(dt)
cdt = 5e3 / np.sqrt(g * max(b.dat.data))
try:
    assert dt < cdt
except:
    print 'WARNING: chosen timestep dt =', dt, 'exceeds recommended value of', cdt
ndump = int(60. / dt)
timings = {}

# Convert pressure gauge locations and save in a dictionary:
glatlon = {'P02': (38.5, 142.5), 'P06': (38.7, 142.6)}
gloc = {}
for key in glatlon:
    east, north, zn, zl = from_latlon(glatlon[key][0], glatlon[key][1], force_zone_number=54)
    gloc[key] = (east, north)
try:
    gauge = raw_input('Gauge P02 or P06?: ') or 'P02'
    gcoord = gloc[gauge]
except:
    ValueError('Gauge not recognised. Please choose P02 or P06.')

# Evaluate and plot (dimensionless) Coriolis parameter function:
f = Function(W.sub(2), name='Coriolis parameter')
for i in range(len(coords)):
    if coords[i][0] < 100000:
        f.dat.data[i] = 2 * Om * sin(radians(get_latitude(100000, coords[i][1], 54, northern=True)))
    elif coords[i][0] < 999999:
        f.dat.data[i] = 2 * Om * sin(radians(get_latitude(coords[i][0], coords[i][1], 54, northern=True)))
    else:
        f.dat.data[i] = 2 * Om * sin(radians(get_latitude(999999, coords[i][1], 54, northern=True)))
File('plots/tsunami_outputs/Coriolis_parameter.pvd').write(f)

# Specify solver parameters:
params = {'mat_type': 'matfree',
          'snes_type': 'ksponly',
          'pc_type': 'python',
          'pc_python_type': 'firedrake.AssembledPC',
          'assembled_pc_type': 'lu',
          'snes_lag_preconditioner': -1,
          'snes_lag_preconditioner_persists': True}

# params = {'ksp_type': 'gmres',
#           'ksp_rtol': '1e-8',
#           'pc_type': 'fieldsplit',
#           'pc_fieldsplit_type': 'schur',
#           'pc_fieldsplit_schur_fact_type': 'full',
#           'fieldsplit_0_ksp_type': 'cg',
#           'fieldsplit_0_pc_type': 'ilu',
#           'fieldsplit_1_ksp_type': 'cg',
#           'fieldsplit_1_pc_type': 'hypre',
#           'pc_fieldsplit_schur_precondition': 'selfp',}

for key in mode:
    print ''
    print '****************', mode[key], ' case (', key, ') ****************'
    print ''

    # Assign initial surface and post-process the bathymetry to have a minimum depth of 30m:
    u_.interpolate(Expression(0))
    v_.interpolate(Expression(0))
    eta_.assign(eta0)

    # Set up functions of the weak problem:
    q = Function(W)
    q.assign(q_)
    w, z, ze = TestFunctions(W)
    u, v, eta = split(q)
    u_, v_, eta_ = split(q_)

    L = (ze * (eta - eta_) - Dt * b * (u * ze.dx(0) + v * ze.dx(1))
         + (u - u_) * w + Dt * g * eta.dx(0) * w
         + (v - v_) * z + Dt * g * eta.dx(1) * z) * dx
    if key in (1, 3):                                   # Rotational cases
        L += Dt * f * (u * z - v * w) * dx
    if key in (2, 3):                                   # Nonlinear cases
        L += Dt * (- eta * (u * ze.dx(0) + v * ze.dx(1))
                   + (u * u.dx(0) + v * u.dx(1)) * w + (u * v.dx(0) + v * v.dx(1)) * z
                   + nu * (u.dx(0) * w.dx(0) + u.dx(1) * w.dx(1) + v.dx(0) * z.dx(0) + v.dx(1) * z.dx(1))
                   + Cb * sqrt(u_ * u_ + v_ * v_) * (u * w + v * z) / (eta + b)) * dx
    q_prob = NonlinearVariationalProblem(L, q)
    q_solv = NonlinearVariationalSolver(q_prob, solver_parameters=params)

    # 'Split' functions in order to access their data and then relabel:
    u_, v_, eta_ = q_.split()
    u, v, eta = q.split()
    u.rename('Fluid x-velocity')
    v.rename('Fluid y-velocity')
    eta.rename('Free surface displacement')

    # Initialise counters and files:
    t = 0.
    dumpn = 0
    q_file = File('plots/tsunami_outputs/model_verif_{y}.pvd'.format(y=key))
    q_file.write(u, v, eta, time=t)
    gauge_dat = [eta.at(gcoord)]
    tic1 = clock()

    while t < T - 0.5 * dt:
        t += dt
        print 't = %1.1f mins' % (t / 60.)
        q_solv.solve()
        q_.assign(q)
        dumpn += 1
        gauge_dat.append(eta.at(gcoord))
        if dumpn == ndump:
            dumpn -= ndump
            q_file.write(u, v, eta, time=t)

    # End timing and print:
    toc1 = clock()
    timings[key] = toc1 - tic1

    # Plot pressure gauge time series:
    plt.rc('text', usetex=True)
    font = {'family': 'serif',
            'size': 18}
    plt.rc('font', **font)
    plt.plot(np.linspace(0, 60, len(gauge_dat)), gauge_dat, label=mode[key])
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([-5, 5])
    # plt.legend()
    plt.xlabel(r'Time elapsed (mins)')
    plt.ylabel(r'Free surface (m)')
print ''
plt.savefig('plots/tsunami_outputs/screenshots/gauge_timeseries_{y}.png'.format(y=gauge))

for key in mode:
    print mode[key], 'case time =  %1.1fs' % timings[key]
