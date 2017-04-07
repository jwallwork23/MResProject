from firedrake import *
from thetis import *
import scipy.interpolate as si
from scipy.io.netcdf import NetCDFFile
from math import radians
import numpy as np 

######################## CONVERSION FUNCTIONS #########################

def earth_radius(lat):
    '''A function which calculates the radius of the Earth for a given
    latitude.'''
    K = 1./298.257  # Earth flatness constant
    a = 6378136.3   # Semi-major axis of the Earth (m)
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

def mesh_converter(meshfile):
    '''A function which reads in a .msh file in longitude-latitude
    coordinates and outputs a tangent-plane projection in
    Cartesian coordinates.'''
    mesh1 = open(meshfile, 'r')
    mesh2 = open('meshes/CartesianTohoku.msh', 'w')
    i = 0
    mode = 0
    cnt = 0
    N = -1
    for line in mesh1:
        i += 1
        if (i == 5):
            mode += 1
        if (mode == 1):
            N = int(line)
            mode += 1
        elif (mode == 2):
            xy = line.split()
            xy[1], xy[2] = lonlat2tangentxy(float(xy[1]), \
                                            float(xy[2]), 143., 37.)
            xy[1] = str(xy[1])
            xy[2] = str(xy[2])
            line = ' '.join(xy)
            line += '\n'
            cnt += 1
        if (cnt == N):
            mode += 1
            cnt +=1
        mesh2.write(line)
    mesh1.close()
    mesh2.close()
    
########################### USER INPUT ################################

# Specify problem parameters:
mode = raw_input('Use linear or nonlinear equations? (l/n): ') or 'l'
if ((mode != 'l') & (mode != 'n')):
    raise ValueError('Please try again, choosing l or n.')

# Specify time parameters:
dt = float(raw_input('Specify timestep (s) (default 15):') or 15)
Dt = Constant(dt)
ndump = 4
t_export = ndump * dt
T = float(raw_input('Specify time period (s) (default 7200):') or 7200)
# INCLUDE FUNCTIONALITY FOR MESH CHOICE

############################ FE SETUP #################################

# Set physical and numerical parameters for the scheme:
nu = 1e-3           # Viscosity
g = 9.81            # Gravitational acceleration
Cb = 0.0025         # Bottom friction coefficient
depth = 0.1         # Specify tank water depth

# Define mesh (courtesy of QMESH) and function spaces:
mesh_converter('meshes/tohoku_edit.msh')
mesh = Mesh('meshes/CartesianTohoku.msh')     # Japanese coastline
mesh_coords = mesh.coordinates.dat.data
Vu = VectorFunctionSpace(mesh, 'CG', 2) # \ Use Taylor-Hood elements
Ve = FunctionSpace(mesh, 'CG', 1)       # /
Vq = MixedFunctionSpace((Vu, Ve))       # We have a mixed FE problem

# Construct functions to store dependent variables and bathymetry:
q_ = Function(Vq)  
u_, eta_ = q_.split()
b = Function(Vq.sub(1), name='Bathymetry')   # Bathymetry function

################## INITIAL AND BOUNDARY CONDITIONS ####################

# Read and interpolate initial surface data (courtesy of Saito):
nc1 = NetCDFFile('Saito_files/init_profile.nc', mmap=False)
lon1 = nc1.variables['x'][:]
lat1 = nc1.variables['y'][:]
x1, y1 = vectorlonlat2tangentxy(lon1, lat1, 143., 37.)
elev1 = nc1.variables['z'][:,:]
interpolator_surf = si.RectBivariateSpline(y1, x1, elev1)
eta_vec = eta_.dat.data
assert mesh_coords.shape[0]==eta_vec.shape[0]

# Read and interpolate bathymetry data (courtesy of GEBCO):
nc2 = NetCDFFile('bathy_data/GEBCO_bathy.nc', mmap=False)
lon2 = nc2.variables['lon'][:]
lat2 = nc2.variables['lat'][:-1]
x2, y2 = vectorlonlat2tangentxy(lon2, lat2, 143., 37.)
elev2 = nc2.variables['elevation'][:-1,:]
interpolator_bath = si.RectBivariateSpline(y2, x2, elev2)
b_vec = b.dat.data
assert mesh_coords.shape[0]==b_vec.shape[0]

# Interpolate data onto initial surface and bathymetry profiles:
for i,p in enumerate(mesh_coords):
    eta_vec[i] = interpolator_surf(p[1], p[0])
    b_vec[i] = - interpolator_surf(p[1], p[0]) - \
               interpolator_bath(p[1], p[0])

# Post-process the bathymetry to have minimum depth of 30m:
b.assign(conditional(lt(30, b), b, 30))

# Plot initial surface and bathymetry profiles:
ufile = File('tsunami_outputs/init_surf.pvd')
ufile.write(eta_)
ufile = File('tsunami_outputs/tsunami_bathy.pvd')
ufile.write(b)

########################## WEAK PROBLEM ###############################

# Build the weak form of the timestepping algorithm, expressed as a 
# mixed nonlinear problem:
v, ze = TestFunctions(Vq)
q = Function(Vq)
q.assign(q_)
u, eta = split(q)       # \ Here split means we split up a function so
u_, eta_ = split(q_)    # / it can be inserted into a UFL expression

# Establish form:

def nonlinear_form():
    L = (
     (ze * (eta-eta_) - Dt * inner((eta + b) * u, grad(ze)) + \
      inner(u-u_, v) + Dt * inner(dot(u, nabla_grad(u)), v) + \
      nu * inner(grad(u), grad(v)) + Dt * g * inner(grad(eta), v) + \
      Dt * Cb * sqrt(dot(u_, u_)) * inner(u/(eta+b), v)) * dx(degree=4)   
     )
    return L

def linear_form():
    L = (
    (ze * (eta-eta_) - Dt * inner((eta + b) * u, grad(ze)) + \
    inner(u-u_, v) + Dt * g *(inner(grad(eta), v))) * dx
    )
    return L

if (mode == 'l'):
    L = linear_form()
elif (mode == 'n'):
    L = nonlinear_form()

# Set up the variational problem
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

# The function 'split' has two forms: now use the form which splits a 
# function in order to access its data
u_, eta_ = q_.split()
u, eta = q.split()

############################ TIMESTEPPING #############################

# Store multiple functions
u.rename('Fluid velocity')
eta.rename('Free surface displacement')

# Initialise arrays, files and dump counter
if (mode == 'l'):
    ufile = File('tsunami_outputs/tohoku_linear.pvd')
elif (mode == 'n'):
    ufile = File('tsunami_outputs/tohoku_nonlinear.pvd')
t = 0.0
ufile.write(u, eta, time=t)
ndump = 10
dumpn = 0

# Create a dictionary containing checkpointed values of eta:
checks ={0.0: eta}

# Enter the timeloop:
while (t < T - 0.5*dt):     
    t += dt
    print 't = ', t/60, ' mins'
    ## CALCULATE log_2(eta_max) to evaluate damage at coast
    usolver.solve()
    q_.assign(q)
    dumpn += 1              # Dump the data
    if dumpn == ndump:
        dumpn -= ndump
        ufile.write(u, eta, time=t)
        # TODO: MAKE THIS MORE GENERAL
        checks[float(int(20*t))/20.0 + 0.05] = eta

print len(checks.keys())    # Sanity check

############################ THETIS SETUP #############################

# Construct solver:
solver_obj = solver2d.FlowSolver2d(mesh, b)
options = solver_obj.options
options.t_export = t_export
options.t_end = T
options.timestepper_type = 'backwardeuler'  # Use implicit timestepping
options.dt = dt
options.outputdir = 'tsunami_outputs'

# Specify initial surface elevation:
solver_obj.assign_initial_conditions(elev=eta_)

# Run the model:
solver_obj.iterate()

# OUTPUT CHECKS FOR THETIS TOO

########################### EVALUATE ERROR ############################

##for keys in checks:
    # TO DO
