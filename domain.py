from firedrake import *

depth = 0.1 # Tank water depth (m)

def tank_domain_wp(n):
    '''A function which sets up a uniform mesh and associated functions for the tank test problem. This version is working progress.'''

    # Define domain and mesh:
    lx = 4; ly = 1; nx = lx*n; ny = ly*n
    mesh = RectangleMesh(nx, ny, lx, ly)
    x = SpatialCoordinate(mesh)

    # Define function spaces:
    Vu = VectorFunctionSpace(mesh, 'CG', 1)     # \ TODO: Use Taylor-Hood elements
    Ve = FunctionSpace(mesh, 'CG', 1)           # / 
    Vq = MixedFunctionSpace((Vu, Ve))           # Mixed FE problem

    # Construct a function to store our two variables at time n:
    q_ = Function(Vq)       # \ Split means we can interpolate the initial condition onto the two components 
    u_, eta_ = q_.split()   # / 

    # Construct a (constant) bathymetry function:
    b = Function(Ve, name = 'Bathymetry')
    b.assign(depth)

    # Interpolate ICs:
    u_.interpolate(Expression([0, 0]))
    eta_.interpolate(-0.01*cos(0.5*pi*x[0]))

    return mesh, Vq, q_, u_, eta_, b
