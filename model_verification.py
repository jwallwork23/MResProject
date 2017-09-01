from firedrake import *
import math
import numpy as np
import scipy.interpolate as si
from scipy.io.netcdf import NetCDFFile
from time import clock

from utils.conversion import from_latlon, get_latitude
from utils.domain import Tohoku_domain
from utils.storage import gauge_timeseries

# Change backend to resolve framework problems:
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

print('\n******************************** MODEL VERIFICATION ********************************\nOptions...')

# Establish cases:
mode = {0: 'Linear, non-rotational', 1: 'Linear, rotational',
        2: 'Nonlinear, non-rotational', 3: 'Nonlinear, rotational'}
for key in mode:
    print(key, ' : ', mode[key])
print('\n')
choices = int(raw_input('Choose mode (0/1/2/3 or 4 to try all): ') or 4)
if choices in (0, 1, 2, 3):
    mode = {choices: mode[choices]}
coarseness = int(raw_input('Mesh coarseness? (Integer in range 1-5, default 3): ') or 3)

# Define mesh, mixed function space and variables:
mesh, W, q_, u_, v_, eta_, lam_, lu_, lv_, b = Tohoku_domain(res=coarseness, split=True)
eta0 = Function(W.sub(2), name='Initial free surface')
eta0.assign(eta_)
coords = mesh.coordinates.dat.data
print('........ mesh loaded. Number of vertices : ', len(mesh.coordinates.dat.data))

# Set physical parameters:
Om = 7.291e-5                   # Rotation rate of Earth (rad s^{-1})
g = 9.81                        # Gravitational acceleration (m s^{-2})
nu = 1e-3                       # Viscosity (kg s^{-1} m^{-1})
Cb = 0.0025                     # Bottom friction coefficient (dimensionless)

# Simulation duration:
T = float(raw_input('Simulation duration in hours (default 1)?: ') or 1.) * 3600.
dt = float(raw_input('Specify timestep in seconds (default 1): ') or 1.)
Dt = Constant(dt)
cdt = 5e2 / np.sqrt(g * max(b.dat.data))
if dt > cdt:
    print('WARNING: chosen timestep dt =', dt, 'exceeds recommended value of', cdt)
    if raw_input('Are you happy to proceed? (y/n)') == 'n':
        exit(23)
ndump = int(60. / dt)
timings = {}

# Convert gauge locations to UTM coordinates:
glatlon = {'P02': (38.5002, 142.5016), 'P06': (38.6340, 142.5838),
           '801': (38.2, 141.7), '802': (39.3, 142.1), '803': (38.9, 141.8), '804': (39.7, 142.2), '806': (37.0, 141.2)}
gloc = {}
for key in glatlon:
    east, north, zn, zl = from_latlon(glatlon[key][0], glatlon[key][1], force_zone_number=54)
    gloc[key] = (east, north)

# Set gauge arrays:
gtype = raw_input('Pressure or tide gauge? (p/t): ') or 'p'
if gtype == 'p':
    gauge = raw_input('Gauge P02 or P06? (default P02): ') or 'P02'
    gcoord = gloc[gauge]
elif gtype == 't':
    gauge = raw_input('Gauge 801, 802, 803, 804 or 806? (default 801): ') or '801'
    gcoord = gloc[gauge]
else:
    ValueError('Gauge type not recognised. Please choose p or t.')
dm = raw_input('Evaluate damage measures? (y/n, default n): ') or 'n'

tic1 = clock()
# Evaluate and plot (dimensionless) Coriolis parameter function:
if mode not in (0, 2):
    f = Function(W.sub(2), name='Coriolis parameter')
    for i in range(len(coords)):
        if coords[i][0] < 100000:
            f.dat.data[i] = 2 * Om * sin(math.radians(get_latitude(100000, coords[i][1], 54, northern=True)))
        elif coords[i][0] < 999999:
            f.dat.data[i] = 2 * Om * sin(math.radians(get_latitude(coords[i][0], coords[i][1], 54, northern=True)))
        else:
            f.dat.data[i] = 2 * Om * sin(math.radians(get_latitude(999999, coords[i][1], 54, northern=True)))
    File('plots/tsunami_outputs/Coriolis_parameter.pvd').write(f)

for key in mode:
    print('\n****************', mode[key], ' case (', key, ') ****************\n')

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
    q_solv = NonlinearVariationalSolver(q_prob, solver_parameters={'mat_type': 'matfree',
                                                                   'snes_type': 'ksponly',
                                                                   'pc_type': 'python',
                                                                   'pc_python_type': 'firedrake.AssembledPC',
                                                                   'assembled_pc_type': 'lu',
                                                                   'snes_lag_preconditioner': -1,
                                                                   'snes_lag_preconditioner_persists': True})
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

    if dm == 'y':
        tide0 = {}
        for gtide in ('801', '802', '803', '804', '806'):
            tide0[gtide] = eta.at(gloc[gtide])
        damage_measure = [-1]

    while t < T - 0.5 * dt:
        tic2 = clock()

        # Increment counters:
        t += dt
        dumpn += 1
        print('t = %1.1f mins' % (t / 60.))

        # Solve problem:
        q_solv.solve()
        q_.assign(q)

        # Store data:
        gauge_dat.append(eta.at(gcoord))
        if (key == 0) & (dm == 'y'):
            damage_measure.append(math.log(max(eta.at(gloc['801']) - tide0['801'], eta.at(gloc['802']) - tide0['802'],
                                               eta.at(gloc['803']) - tide0['803'], eta.at(gloc['804']) - tide0['804'],
                                               eta.at(gloc['806']) - tide0['806'], 0.5), 2))
        if dumpn == ndump:
            dumpn -= ndump
            q_file.write(u, v, eta, time=t)
        toc2 = clock()
        duration = toc2 - tic2
        if duration < 60:
            print('[Real time this timestep:', duration, 'seconds]')
        elif duration < 3600:
            print('[Real time this timestep:', (duration / 60.), 'minutes]')
        else:
            print('[Real time this timestep:', (duration / 3600.), 'hours]')

    # End timing and print:
    toc1 = clock()
    timings[key] = toc1 - tic1

    # Plot gauge time series:
    plt.rc('text', usetex=True)
    font = {'family': 'serif',
            'size': 18}
    plt.rc('font', **font)
    plt.plot(np.linspace(0, 60, len(gauge_dat)), gauge_dat, label=mode[key])
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.xlabel(r'Time elapsed (mins)')
    plt.ylabel(r'Free surface (m)')
print('\a')
plt.savefig('plots/tsunami_outputs/screenshots/gauge_timeseries_{y1}_res{y2}.png'.format(y1=gauge, y2=coarseness))

# Store gauge timeseries data to file:
gauge_timeseries(gauge, gauge_dat)

# Plot damage measures time series:
if dm == 'y':
    plt.clf()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.plot(np.linspace(0, 60, len(damage_measure)), damage_measure)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.axis([0, 60, -1.5, 3.5])
    plt.axhline(-1, linestyle='--', color='blue')
    plt.axhline(0, linestyle='--', color='green')
    plt.axhline(1, linestyle='--', color='yellow')
    plt.axhline(2, linestyle='--', color='orange')
    plt.axhline(3, linestyle='--', color='red')
    plt.xlabel(r'Time elapsed (mins)')
    plt.ylabel(r'Maximal log free surface')
    plt.savefig('plots/tsunami_outputs/screenshots/damage_measure_timeseries.png')

for key in mode:
    print(mode[key], 'case time =  %1.1fs' % timings[key])
