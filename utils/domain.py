from firedrake import *
import scipy.interpolate as si
from scipy.io.netcdf import NetCDFFile

from conversion import vectorlonlat2utm


def domain_1d(n):
    """
    Set up a uniform mesh and associated functions for the 1D tsunami test problem.
    
    :param n: number of gridpoints per km.
    :return: mesh, mixed function space forward and adjoint variables and bathymetry field associated with the 1D 
    domain.
    """
    
    # Define domain and mesh:
    lx = 4e5
    nx = int(lx * n)                                                        # 400 km ocean domain, uniform grid spacing
    mesh = IntervalMesh(nx, lx)

    # Define Taylor-Hood mixed function space:
    Vq = FunctionSpace(mesh, 'CG', 2) * FunctionSpace(mesh, 'CG', 1)

    # Construct functions to store forward and adjoint variables:
    q_ = Function(Vq)                                                       # Forward solution tuple
    lam_ = Function(Vq)                                                     # Adjoint solution tuple
    mu_, eta_ = q_.split()                                                  # \ Split means we can interpolate the
    lm_, le_ = lam_.split()                                                 # / initial condition into the components

    # Interpolate initial conditions:
    mu_.interpolate(Expression(0.))
    eta_.interpolate(Expression('(x[0] >= 100e3) & (x[0] <= 150e3) ? 0.4 * sin(pi * (x[0] - 100e3) / 50e3) : 0.0'))

    # Interpolate final-time conditions:
    lm_.interpolate(Expression(0.))
    le_.interpolate(Expression('(x[0] >= 10e3) & (x[0] <= 25e3) ? 0.4 : 0.0'))

    # Interpolate bathymetry:
    b = Function(Vq.sub(1), name='Bathymetry')
    b.interpolate(Expression('x[0] <= 50000. ? 200. : 4000.'))

    return mesh, Vq, q_, mu_, eta_, lam_, lm_, le_, b


def square_domain(n):
    """
    Set up a mesh, along with function spaces and functions, for a 4m x 4m domain.
    
    :param n: number of elements per metre.
    :return: associated mesh, mixed function space forward and adjoint variables and bathymetry field.
    """

    lx = 4                                      # Extent in x- and y-directions (m)
    mesh = SquareMesh(lx * n, lx * n, lx, lx)
    depth = 0.1                                 # Water depth for flat bathymetry case (m)

    # Define mixed Taylor-Hood function space:
    W = VectorFunctionSpace(mesh, 'CG', 2) * FunctionSpace(mesh, 'CG', 1)

    # Repeat above setup:
    q_ = Function(W)
    lam_ = Function(W)
    u_, eta_ = q_.split()
    lu_, le_ = lam_.split()

    # Interpolate initial conditions:
    u_.interpolate(Expression([0, 0]))
    eta_.interpolate(1e-3 * exp(- (pow(x - 2., 2) + pow(y - 2., 2)) / 0.04))
    lu_.interpolate(Expression([0, 0]))
    le_.assign(0)

    # Interpolate bathymetry:
    b = Function(W.sub(1), name='Bathymetry')
    if bathy == 'f':
        b.interpolate(Expression(depth))
    else:
        b.interpolate(Expression('x[0] <= 0.5 ? 0.01 : 0.1'))  # Shelf break bathymetry

    return mesh, W, q_, u_, eta_, lam_, lu_, le_, b


def Tohoku_domain(res=3, split=False):
    """
    Set up a mesh, along with function spaces and functions, for the 2D ocean domain associated with the Tohoku tsunami 
    problem.
    
    :param res: mesh resolution value, ranging from 'extra coarse' (1) to extra fine (5).
    :param split: choose whether to consider the velocity space as vector P2 or as a pair of scalar P2 spaces.
    :return: associated mesh, mixed function space forward and adjoint variables and bathymetry field. 
    """

    # Define mesh and function spaces:
    if res == 1:
        mesh = Mesh('resources/meshes/TohokuXFine.msh')     # 226,967 vertices, ~45 seconds per timestep
        print('WARNING: chosen mesh resolution can be extremely computationally intensive')
        if raw_input('Are you happy to proceed? (y/n)') == 'n':
            exit(23)
    elif res == 2:
        mesh = Mesh('resources/meshes/TohokuFine.msh')      # 97,343 vertices, ~1 second per timestep
    elif res == 3:
        mesh = Mesh('resources/meshes/TohokuMedium.msh')    # 25,976 vertices, ~0.25 seconds per timestep
    elif res == 4:
        mesh = Mesh('resources/meshes/TohokuCoarse.msh')    # 7,194 vertices, ~0.07 seconds per timestep
    elif res == 5:
        mesh = Mesh('resources/meshes/TohokuXCoarse.msh')   # 3,126 vertices, ~0.03 seconds per timestep
    else:
        raise ValueError('Please try again, choosing an integer in the range 1-5.')
    mesh_coords = mesh.coordinates.dat.data

    if split:
        # Define Taylor-Hood mixed function space:
        W = FunctionSpace(mesh, 'CG', 2) * FunctionSpace(mesh, 'CG', 2) * FunctionSpace(mesh, 'CG', 1)

        # Construct functions to store forward and adjoint variables, along with bathymetry:
        q_ = Function(W)
        lam_ = Function(W)
        u_, v_, eta_ = q_.split()
        lu_, lv_, le_ = lam_.split()
        eta0 = Function(W.sub(2), name='Initial surface')
        b = Function(W.sub(2), name='Bathymetry')

        # Specify zero initial fluid velocity:
        u_.interpolate(Expression(0))
        v_.interpolate(Expression(0))
        lu_.interpolate(Expression(0))
        lv_.interpolate(Expression(0))
    else:
        # Define Taylor-Hood mixed function space:
        W = VectorFunctionSpace(mesh, 'CG', 2) * FunctionSpace(mesh, 'CG', 1)

        # Construct functions to store forward and adjoint variables, along with bathymetry:
        q_ = Function(W)
        lam_ = Function(W)
        u_, eta_ = q_.split()
        lu_, le_ = lam_.split()
        eta0 = Function(W.sub(1), name='Initial surface')
        b = Function(W.sub(1), name='Bathymetry')

        # Specify zero initial fluid velocity:
        u_.interpolate(Expression([0, 0]))
        lu_.interpolate(Expression([0, 0]))


    # Read and interpolate initial surface data (courtesy of Saito):
    nc1 = NetCDFFile('resources/Saito_files/init_profile.nc', mmap=False)
    lon1 = nc1.variables['x'][:]
    lat1 = nc1.variables['y'][:]
    x1, y1 = vectorlonlat2utm(lat1, lon1, force_zone_number=54)             # Our mesh mainly resides in UTM zone 54
    elev1 = nc1.variables['z'][:, :]
    interpolator_surf = si.RectBivariateSpline(y1, x1, elev1)
    eta0vec = eta0.dat.data
    assert mesh_coords.shape[0] == eta0vec.shape[0]

    # Read and interpolate bathymetry data (courtesy of GEBCO):
    nc2 = NetCDFFile('resources/bathy_data/GEBCO_bathy.nc', mmap=False)
    lon2 = nc2.variables['lon'][:]
    lat2 = nc2.variables['lat'][:-1]
    x2, y2 = vectorlonlat2utm(lat2, lon2, force_zone_number=54)
    elev2 = nc2.variables['elevation'][:-1, :]
    interpolator_bath = si.RectBivariateSpline(y2, x2, elev2)
    b_vec = b.dat.data
    assert mesh_coords.shape[0] == b_vec.shape[0]

    # Interpolate data onto initial surface and bathymetry profiles:
    for i, p in enumerate(mesh_coords):
        eta0vec[i] = interpolator_surf(p[1], p[0])
        b_vec[i] = - interpolator_surf(p[1], p[0]) - interpolator_bath(p[1], p[0])

    # Assign initial surface and post-process the bathymetry to have a minimum depth of 30m:
    eta_.assign(eta0)
    le_.assign(0)
    b.assign(conditional(lt(30, b), b, 30))

    # Plot initial surface and bathymetry profiles:
    File('plots/tsunami_outputs/init_surf.pvd').write(eta0)
    File('plots/tsunami_outputs/tsunami_bathy.pvd').write(b)

    if split:
        return mesh, W, q_, u_, v_, eta_, lam_, lu_, lv_, b
    else:
        return mesh, W, q_, u_, eta_, lam_, lu_, le_, b
