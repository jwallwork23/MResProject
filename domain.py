from firedrake import *

depth = 0.1 # Tank water depth (m)

def tank_domain(n, bath='n', waves='n'):
    '''A function which sets up a uniform mesh and associated functions for the tank test problem.'''

    # Define domain and mesh:
    lx = 4; ly = 1; nx = lx*n; ny = ly*n
    mesh = RectangleMesh(nx, ny, lx, ly)
    x = SpatialCoordinate(mesh)

    # Define function spaces:
    Vu = VectorFunctionSpace(mesh, 'CG', 2) # \ Taylor-Hood elements
    Ve = FunctionSpace(mesh, 'CG', 1)       # /
    Vq = MixedFunctionSpace((Vu, Ve))       # Mixed FE problem

    # Construct a function to store our two variables at time n:
    q_ = Function(Vq)       # \ Split means we can interpolate the initial condition onto the two components 
    u_, eta_ = q_.split()   # / 

    # Construct a (constant) bathymetry function:
    b = Function(Ve, name = 'Bathymetry')
    if (bath == 'n'):
        # Construct a (constant) bathymetry function:
        b.assign(depth)
    elif (bath == 'y'):
        b.interpolate(0.1 + 0.04 * sin(2*pi*x[0]) * sin(2*pi*x[1]))
        File('plots/screenshots/tank_bathymetry.pvd').write(b)

    # Interpolate ICs:
    u_.interpolate(Expression([0, 0]))
    if (waves == 'n'):
        eta_.interpolate(-0.01*cos(0.5*pi*x[0]))
    else:
        eta_.interpolate(Expression(0))
        # Establish a BC object for the oscillating inflow condition:
        bcval = Constant(0.0)
        bc1 = DirichletBC(Vq.sub(1), bcval, 1)
        # Apply no-slip BC to eta on the right end of the domain:
        bc2 = DirichletBC(Vq.sub(1), (0.0), 2)

    return mesh, Vq, q_, u_, eta_, b
