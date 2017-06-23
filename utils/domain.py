from firedrake import *
import numpy as np

from projection import *

def domain_1d(n):
    """A function which sets up a uniform mesh and associated functions for the 1D tsunami test problem."""
    
    # Define domain and mesh:
    lx = 4e5
    nx = int(lx * n)    # 400 km ocean domain, uniform grid spacing
    mesh = IntervalMesh(nx, lx)

    # Define function spaces:
    Vmu = FunctionSpace(mesh, 'CG', 2)                                      # \ Use Taylor-Hood elements
    Ve = FunctionSpace(mesh, 'CG', 1)                                       # /
    Vq = MixedFunctionSpace((Vmu, Ve))                                      # We have a mixed FE problem

    # Construct functions to store forward and adjoint variables:
    q_ = Function(Vq)                                                       # Forward solution tuple
    lam_ = Function(Vq)                                                     # Adjoint solution tuple
    mu_, eta_ = q_.split()                                                  # \ Split means we can interpolate the
    lm_, le_ = lam_.split()                                                 # / initial condition into the components

    # Interpolate initial conditions:
    mu_.interpolate(Expression(0.))
    eta_.interpolate(Expression('(x[0] >= 1e5) & (x[0] <= 1.5e5) ? 0.4*sin(pi*(x[0]-1e5)/5e4) : 0.0'))

    # Interpolate final-time conditions:
    lm_.interpolate(Expression(0.))
    le_.interpolate(Expression('(x[0] >= 1e4) & (x[0] <= 2.5e4) ? 0.4 : 0.0'))

    # Interpolate bathymetry:
    b = Function(Ve, name = 'Bathymetry')
    b.interpolate(Expression('x[0] <= 50000.0 ? 200.0 : 4000.0'))

    return mesh, Vq, q_, mu_, eta_, lam_, lm_, le_, b


def tank_domain(n, bath = 'n', waves = 'n', test2d = 'n', bcval = None) :
    """A function which sets up a uniform mesh and associated functions for the tank test problem."""

    # Define domain and mesh:
    if test2d == 'n' :
        lx = 4
        ly = 1
        nx = int(lx * n)
        ny = int(ly * n)
        mesh = RectangleMesh(nx, ny, lx, ly)
    else :
        lx = 4e5
        nx = int(lx * n)
        mesh = SquareMesh(nx, nx, lx, lx)
    x = SpatialCoordinate(mesh)

    # Define function spaces:
    Vu = VectorFunctionSpace(mesh, 'CG', 2)                                 # \ Taylor-Hood elements
    Ve = FunctionSpace(mesh, 'CG', 1)                                       # /
    Vq = MixedFunctionSpace((Vu, Ve))                                       # Mixed FE problem

    # Construct a function to store our two variables at time n:
    q_ = Function(Vq)                                                       # Forward solution tuple
    lam_ = Function(Vq)                                                     # Adjoint solution tuple
    u_, eta_ = q_.split()                                                   # \ Split means we can interpolate the
    lu_, le_ = lam_.split()                                                 # / initial condition onto the components

    # Establish bathymetry function:
    b = Function(Ve, name = 'Bathymetry')
    if bath == 'y' :
        b.interpolate(0.1 + 0.04 * sin(2 * pi * x[0]) * sin(2 * pi * x[1]))
        File('plots/screenshots/tank_bathymetry.pvd').write(b)
    elif test2d == 'n' :
        # Construct a (constant) bathymetry function:
        b.assign(0.1)                                                       # Tank water depth 10 cm
    else :
        b.interpolate(Expression('x[0] <= 50000. ? 200. : 4000.'))          # Shelf break bathymetry

    # Interpolate forward and adjoint initial and boundary conditions:
    u_.interpolate(Expression([0, 0]))
    lu_.interpolate(Expression([0, 0]))
    BCs = []
    if waves == 'y' :
        eta_.interpolate(Expression(0))
        bc1 = DirichletBC(Vq.sub(1), bcval, 1)
        # Apply no-slip BC to eta on the right end of the domain:
        bc2 = DirichletBC(Vq.sub(1), 0.0, 2)
        BCs = [bc1, bc2]
    elif test2d == 'n' :
        eta_.interpolate(-0.01 * cos(0.5 * pi * x[0]))
    else:   # NOTE: higher magnitude wave used due to geometric spreading
        eta_.interpolate(Expression('(x[0] >= 1e5) & (x[0] <= 1.5e5) & (x[1] >= 1.8e5) & (x[1] <= 2.2e5) ? \
                                    4 * sin(pi*(x[0]-1e5) * 2e-5) * sin(pi*(x[1]-1.8e5) * 2.5e-5) : 0.'))
        le_.interpolate(Expression('(x[0] >= 1e4) & (x[0] <= 2.5e4) & (x[1] >= 1.8e5) & (x[1] <= 2.2e5) ? '
                                   '4 : 0.'))

    return mesh, Vq, q_, u_, eta_, lam_, lu_, le_, b, BCs

def Tohoku_domain(res = 'c') :
    """A function which sets up a mesh, along with function spaces and functions, for the ocean domain associated
    for the Tohoku tsunami problem."""

    import scipy.interpolate as si
    from scipy.io.netcdf import NetCDFFile

    # Define mesh and function spaces:
    if res == 'f' :
        mesh_converter('resources/meshes/LonLatTohokuFine.msh', 143., 37.)
    elif res == 'm' :
        mesh_converter('resources/meshes/LonLatTohokuMedium.msh', 143., 37.)
    elif res == 'c' :
        mesh_converter('resources/meshes/LonLatTohokuCoarse.msh', 143., 37.)
    mesh = Mesh('resources/meshes/CartesianTohoku.msh')
    mesh_coords = mesh.coordinates.dat.data
    Vu = VectorFunctionSpace(mesh, 'CG', 1)                                 # TODO: Use Taylor-Hood elements
    Ve = FunctionSpace(mesh, 'CG', 1)                                       #
    Vq = MixedFunctionSpace((Vu, Ve))                                       # Mixed FE problem

    # Construct functions to store forward and adjoint variables, along with bathymetry:
    q_ = Function(Vq)
    lam_ = Function(Vq)
    u_, eta_ = q_.split()
    lm_, le_ = lam_.split()
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
    u_.interpolate(Expression([0, 0]))
    lm_.interpolate(Expression([0, 0]))
    eta_.assign(eta0)
    le_.interpolate(Expression('(x[0] > -160e3) & (x[0] < -130e3) & (x[1] > 10e3) & (x[1] < 100e3) ? 20 : 0'))
    b.assign(conditional(lt(30, b), b, 30))

    # Plot initial surface and bathymetry profiles:
    File('plots/tsunami_outputs/init_surf.pvd').write(eta0)
    File('plots/tsunami_outputs/tsunami_bathy.pvd').write(b)

    return mesh, Vq, q_, u_, eta_, lam_, lm_, le_, b
