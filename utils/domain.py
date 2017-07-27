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


def tank_domain(n, bath='n', waves='n', test2d='n', bcval=None):
    """
    Set up a uniform mesh and associated functions for the tank test problem.
    
    :param n: number of cells per m in x- and y-directions.
    :param bath: non-trivial bathymetry option.
    :param waves: 'wave generator' option.
    :param test2d: large scale option.
    :param bcval: boundary condition value specification.
    :return: mesh, mixed function space forward and adjoint variables, bathymetry field and boundary condtions 
    associated with the 2D tank domain.
    """

    # Define domain and mesh:
    if test2d == 'n':
        lx = 4
        ly = 1
        nx = int(lx * n)
        ny = int(ly * n)
        mesh = RectangleMesh(nx, ny, lx, ly)
    else:
        lx = 4e5
        nx = int(lx * n)
        mesh = SquareMesh(nx, nx, lx, lx)
    x = SpatialCoordinate(mesh)

    # Define Taylor-Hood mixed function space:
    Vq = VectorFunctionSpace(mesh, 'CG', 1) * FunctionSpace(mesh, 'CG', 1)

    # Construct a function to store our two variables at time n:
    q_ = Function(Vq)                                                       # Forward solution tuple
    lam_ = Function(Vq)                                                     # Adjoint solution tuple
    u_, eta_ = q_.split()                                                   # \ Split means we can interpolate the
    lu_, le_ = lam_.split()                                                 # / initial condition onto the components

    # Establish bathymetry function:
    b = Function(Vq.sub(1), name='Bathymetry')
    if bath == 'y':
        b.interpolate(0.1 + 0.04 * sin(2 * pi * x[0]) * sin(2 * pi * x[1]))
        File('plots/screenshots/tank_bathymetry.pvd').write(b)
    elif test2d == 'n':
        # Construct a (constant) bathymetry function:
        b.assign(0.1)                                                       # Tank water depth 10 cm
    else:
        b.interpolate(Expression('x[0] <= 50000. ? 200. : 4000.'))          # Shelf break bathymetry

    # Interpolate forward and adjoint initial and boundary conditions:
    u_.interpolate(Expression([0, 0]))
    lu_.interpolate(Expression([0, 0]))
    bcs = []
    if waves == 'y':
        eta_.interpolate(Expression(0))
        bc1 = DirichletBC(Vq.sub(1), bcval, 1)
        # Apply no-slip BC to eta on the right end of the domain:
        bc2 = DirichletBC(Vq.sub(1), 0.0, 2)
        bcs = [bc1, bc2]
    elif test2d == 'n':
        eta_.interpolate(-0.01 * cos(0.5 * pi * x[0]))
    else:                                              # NOTE: higher magnitude wave used due to geometric spreading
        eta_.interpolate(Expression('(x[0] >= 1e5) & (x[0] <= 1.5e5) & (x[1] >= 1.8e5) & (x[1] <= 2.2e5) ? \
                                    4 * sin(pi*(x[0]-1e5) * 2e-5) * sin(pi*(x[1]-1.8e5) * 2.5e-5) : 0.'))
        le_.interpolate(Expression('(x[0] >= 1e4) & (x[0] <= 2.5e4) & (x[1] >= 1.8e5) & (x[1] <= 2.2e5) ? 1. : 0.'))

    return mesh, Vq, q_, u_, eta_, lam_, lu_, le_, b, bcs


def Tohoku_domain(res=3, split='n'):
    """
    Set up a mesh, along with function spaces and functions, for the ocean domain associated
    for the Tohoku tsunami problem.
    
    :param res: mesh resolution value, ranging from 'extra coarse' (1) to extra fine (5).
    :param split: choose whether to consider the velocity space as vector P2 or as a pair of scalar P2 spaces.
    :return: mesh, mixed function space forward and adjoint variables and bathymetry field associated with the 2D 
    ocean domain.
    """

    # Define mesh and function spaces:
    if res == 1:
        mesh = Mesh('resources/meshes/TohokuXFine.msh')
    elif res == 2:
        mesh = Mesh('resources/meshes/TohokuFine.msh')
    elif res == 3:
        mesh = Mesh('resources/meshes/TohokuMedium.msh')
    elif res == 4:
        mesh = Mesh('resources/meshes/TohokuCoarse.msh')
    elif res == 5:
        mesh = Mesh('resources/meshes/TohokuXCoarse.msh')
    else:
        raise ValueError('Please try again, choosing an integer in the range 1-5.')
    mesh_coords = mesh.coordinates.dat.data

    if split == 'n':
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

    elif split == 'y':
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
        raise ValueError('Please try again, split equalling y or n.')

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
    le_.interpolate(Expression('(x[0] > 490e3) & (x[0] < 580e3) & (x[1] > 4130e3) & (x[1] < 4260e3) ? 1. : 0.'))
    b.assign(conditional(lt(30, b), b, 30))

    # Plot initial surface and bathymetry profiles:
    File('plots/tsunami_outputs/init_surf.pvd').write(eta0)
    File('plots/tsunami_outputs/tsunami_bathy.pvd').write(b)

    if split == 'n':
        return mesh, W, q_, u_, eta_, lam_, lu_, le_, b
    else:
        return mesh, W, q_, u_, v_, eta_, lam_, lu_, lv_, b
