from firedrake import *
import numpy as np
import Okada_b2av_Dynamic as okada
import scipy.interpolate
from scipy.io.netcdf import NetCDFFile
import GFD_basisChange_tools as gfd
import utm

### FE SETUP ###

# Establish dimensional scales
Lx = 93453.18	# 1 deg. longitude (m) at 33N (hor. length scale)
Ly = 110904.44	# 1 deg. latitude (m) at 33N (ver. length scale)
Lm = 1/sqrt(Lx**2 + Ly**2)	# Inverse magnitude of length scales
Ts = 15 	# Quarter minute in s (timescale)
# Mass scale?

# Set physical and numerical parameters for the scheme
nu = 1e-3           # Viscosity (kgs^{-1}m^{-1})
g = 9.81            # Gravitational acceleration (ms^{-2})
Cb = 0.0025         # Bottom friction coefficient (dimensionless)
dt = Ts             # Timestep, chosen small enough for stability (s)
Dt = Constant(dt)
n = 4               # Mesh resolution parameter

# Compute Okada function to obtain fault characteristics
X, Y, Z, Xfbar, Yfbar, Zfbar, sflength, sfwidth = okada.main()
interpolator_surf = scipy.interpolate.RectBivariateSpline(Y, X, Z)

# Define mesh, function spaces and initial surface
mesh = Mesh("1000_5000_50000_coarse.msh")   # Japanese coastline
Vu = VectorFunctionSpace(mesh, "CG", 2)     # \ Use Taylor-Hood elements
Ve = FunctionSpace(mesh, "CG", 1)           # /
W = MixedFunctionSpace((Vu, Ve))

# Construct a function to store our two variables at time n
w_ = Function(W)            # Split means we can interpolate the 
u_, eta_ = w_.split()       # initial condition into the two components

mesh_coords = mesh.coordinates.dat.data
eta_vec = eta_.dat.data
assert mesh_coords.shape[0]==eta_vec.shape[0]

for i,xy in enumerate(mesh_coords):
    lat, lon = utm.to_latlon(xy[0], xy[1], 54, 'S')
    eta_vec[i] = interpolator_surf(lat, lon)

# Plot initial surface
ufile = File('plots/init_surf_test.pvd')
ufile.write(eta_)

# Read bathymetry data
b = -1750   # Average depth of Japan basin

### INITIAL AND BOUNDARY CONDITIONS ###

# Interpolate ICs
u_.interpolate(Expression([0, 0]))

### WEAK PROBLEM ###

# Build the weak form of the timestepping algorithm, expressed as a 
# mixed nonlinear problem
v, xi = TestFunctions(W)
w = Function(W)
w.assign(w_)

# Here we split up a function so it can be inserted into a UFL
# expression
u, eta = split(w)      
u_, eta_ = split(w_)

# Establish the bilinear form - a function of the output function w.
# We use exact integration of degree 4 polynomials used since the 
# problem we consider is not very nonlinear
L = (
    (xi*(eta-eta_) - Lm*Dt*inner((eta+b)*u, grad(xi))\
    + Lm*inner(u-u_, v) + Lm*Lm*2*Dt*(inner(dot(u, nabla_grad(u)), v)\
    + Lm*nu*inner(grad(u), grad(v)) + g*inner(grad(eta), v))\
    + Lm*Lm*Dt*Cb*sqrt(dot(u_, u_))*inner(u/(eta+b), v))*dx(degree=4)   
) 

# Set up the nonlinear problem
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

# The function 'split' has two forms: now use the form which splits a 
# function in order to access its data
u_, eta_ = w_.split()
u, eta = w.split()

### TIMESTEPPING ###

# Store multiple functions
u.rename("Fluid velocity")
eta.rename("Free surface displacement")

# Choose a final time and initialise arrays, files and dump counter
T = 10.0*Ts
ufile = File('plots/tsunami_SW_test.pvd')
t = 0.0
ufile.write(u, eta, time=t)
ndump = 4
dumpn = 0

while (t < T - 0.5*dt):     # Enter the timeloop
    t += dt
    print "t = ", t/60, " mins"
    usolver.solve()
    w_.assign(w)
    dumpn += 1              # Dump the data
    if dumpn == ndump:
        dumpn -= ndump
        ufile.write(u, eta, time=t)
