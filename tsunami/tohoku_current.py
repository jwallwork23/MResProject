from firedrake import *
import numpy as np
import scipy.interpolate
from scipy.io.netcdf import NetCDFFile
import GFD_basisChange_tools as gfd
import utm

############################ USEFUL FUNCTIONS ##################################

# Read initial surface data:
def get_coords():
    with open("Saito_files/init_profile", "r") as f:
        data = f.readlines()
    l = len(data)
    # Dimension preallocated for speed:
    X = np.zeros((l, 1))
    Y = np.zeros((l, 1))
    Z = np.zeros((l, 1))
    i = 0
    for line in data:
        words = line.split()
        X[i] = words[0]         # Longitude data
        Y[i] = words[1]         # Latitude data
        Z[i] = words[2]         # Free surface displacement data
        i += 1
    return X, Y, Z

################################# FE SETUP #####################################

# Establish dimensional scales:
Lx = 93453.18	            # 1 deg. longitude (m) at 33N (hor. length scale)
Ly = 110904.44	            # 1 deg. latitude (m) at 33N (ver. length scale)
Lm = 1/sqrt(Lx**2 + Ly**2)  # Inverse magnitude of length scales
Ts = 15  	            # Quarter minute in s (timescale)
                        # Mass scale?

# Set physical and numerical parameters for the scheme:
nu = 1e-3           # Viscosity (kgs^{-1}m^{-1})
g = 9.81            # Gravitational acceleration (ms^{-2})
Cb = 0.0025         # Bottom friction coefficient (dimensionless)
dt = Ts             # Timestep, chosen small enough for stability (s)
Dt = Constant(dt)

# Define mesh (courtesy of QMESH), function spaces and initial surface:
mesh = Mesh("meshes/point1_point5_point5.msh")     # Japanese coastline
Vu = VectorFunctionSpace(mesh, "CG", 2)     # \ Use Taylor-Hood elements
Ve = FunctionSpace(mesh, "CG", 1)           # /
W = MixedFunctionSpace((Vu, Ve))            # We consider a mixed FE problem

# Construct functions to store dependent variables and bathymetry:
w_ = Function(W)                            # \ Here 'split' means we  
u_, eta_ = w_.split()                       # / interpolate IC into components
b = Function(W.sub(1), name="Bathymetry")   # Bathymetry function

############### INITIAL AND BOUNDARY CONDITIONS AND BATHYMETRY #################

##  # Compute Okada function to obtain fault characteristics:
##  X, Y, Z, Xfbar, Yfbar, Zfbar, sflength, sfwidth = okada.main()
##  interpolator_surf = scipy.interpolate.RectBivariateSpline(Y, X, Z)

# Read and interpolate initial surface data (courtesy of Saito):
X, Y, Z = get_coords()
interpolator_surf = scipy.interpolate.SmoothBivariateSpline(Y, X, Z)
mesh_coords = mesh.coordinates.dat.data
eta_vec = eta_.dat.data
assert mesh_coords.shape[0]==eta_vec.shape[0]

# Read and interpolate bathymetry data (courtesy of GEBCO):
nc = NetCDFFile('bathy_data/GEBCO_bathy.nc')
lon = nc.variables['lon'][:]
lat = nc.variables['lat'][:]
elev = nc.variables['elevation'][:,:]
interpolator_bath = scipy.interpolate.RectBivariateSpline(lat, lon, elev)
b_vec = b.dat.data
assert mesh_coords.shape[0]==b_vec.shape[0]

# Interpolate data onto initial surface and bathymetry profiles:
for i,p in enumerate(mesh_coords):
    eta_vec[i] = interpolator_surf(p[1], p[0])
    b_vec[i] = - interpolator_surf(p[1], p[0]) - interpolator_bath(p[1], p[0])

# Post-process the bathymetry to have minimum depth of 30m:
b.assign(conditional(lt(30, b), b, 30))

# Plot initial surface and bathymetry profiles:
ufile = File('plots/init_surf.pvd')
ufile.write(eta_)
ufile = File('plots/tsunami_bathy.pvd')
ufile.write(b)

# Interpolate IC on fluid velocity:
u_.interpolate(Expression([0, 0]))

################################# WEAK PROBLEM #################################

# Build the weak form of the timestepping algorithm, expressed as a 
# mixed nonlinear problem:
v, xi = TestFunctions(W)
w = Function(W)
w.assign(w_)
u, eta = split(w)           # \ Here split means we split up a function so
u_, eta_ = split(w_)        # / it can be inserted into a UFL expression

# Establish the bilinear form, a function of the output function w.
# (NOTE: We use exact integration of degree 4 polynomials used since the 
# problem we consider is 'not very nonlinear'):
L = (
    (xi*(eta-eta_) - Lm*Dt*inner((eta+b)*u, grad(xi))\
    + Lm*inner(u-u_, v) + Lm*Lm*2*Dt*(inner(dot(u, nabla_grad(u)), v)\
    + Lm*nu*inner(grad(u), grad(v)) + g*inner(grad(eta), v))\
    + Lm*Lm*Dt*Cb*sqrt(dot(u_, u_))*inner(u/(eta+b), v))*dx(degree=4)   
) 

# Set up the nonlinear problem and specify solver parameters:
uprob = NonlinearVariationalProblem(L, w)
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
u_, eta_ = w_.split()                           # IS THIS NEEDED?
u, eta = w.split()

################################# TIMESTEPPING #################################

# Store multiple functions
u.rename("Fluid velocity")
eta.rename("Free surface displacement")

# Choose a final time and initialise arrays, files and dump counter
T = 500.0*Ts
ufile = File('plots/simulation.pvd')
t = 0.0
ufile.write(u, eta, time=t)
ndump = 20
dumpn = 0

while (t < T - 0.5*dt):     # Enter the timeloop
    t += dt
    if (t % 60 == 0):
        print "t = ", t/60, " mins"
    usolver.solve()
    w_.assign(w)
    dumpn += 1              # Dump the data
    if dumpn == ndump:
        dumpn -= ndump
        ufile.write(u, eta, time=t)
