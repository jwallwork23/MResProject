from firedrake import *

### FE SETUP ###

# Set physical and numerical parameters for the scheme
nu = 1e-3           # Viscosity
g = 9.81            # Gravitational acceleration
Cb = 1e-7           # Bottom friction coefficient
dt = 0.01           # Timestep, chosen small enough for stability                          
Dt = Constant(dt)

# Define domain and mesh
n = 30
mesh = RectangleMesh(4*n, n, 4, 1)

# Define function spaces
Vu  = VectorFunctionSpace(mesh, "CG", 2)    # Use Taylor-Hood elements
Ve = FunctionSpace(mesh, "CG", 1)           
W = MixedFunctionSpace((Vu, Ve))

# Set bathymetry
h = Function(Ve)
h.interpolate(Expression('0.1-0.05*x[0]'))

# Construct a function to store our two variables at time n
w0 = Function(W)            # Split means we can interpolate the 
u0, e0 = w0.split()         # initial condition into the two components

# Consider steady or non-steady case
kappa = input("Steady (enter 0 ) or non-steady (enter 1) case?: ")

if (kappa == 1):

    ### INITIAL AND BOUNDARY CONDITIONS ###

    # Interpolate ICs
    u0.interpolate(Expression([0, 0]))
    e0.interpolate(Expression('0.01*sin(x[0])'))

    # Apply no-slip BCs on the top and bottom edges of the domain
    bc1 = DirichletBC(W.sub(0), (0.0,0.0), (3,4))

    ### WEAK PROBLEM ###

    # Build the weak form of the timestepping algorithm, expressed as a 
    # mixed nonlinear problem
    v, xi = TestFunctions(W)
    w1 = Function(W)
    w1.assign(w0)

    # Here we split up a function so it can be inserted into a UFL
    # expression
    u1, e1 = split(w1)      
    u0, e0 = split(w0)

    # Establish the bilinear form - a function of the output function w1
    L = (
        (xi*(e1-e0) - Dt*inner((e1+h)*u1, grad(xi)))*dx\
        + (inner(u1-u0, v) + Dt*(inner(dot(u1, nabla_grad(u1)), v)\
        + nu*inner(grad(u1), grad(v)) + g*inner(grad(e1), v)))*dx
        + Dt*Cb*sqrt(dot(u0,u0))*inner(u1/(e1+h),v)*dx
    )

    # Set up the nonlinear problem
    uprob = NonlinearVariationalProblem(L, w1, bcs=bc1)
    usolver = NonlinearVariationalSolver(uprob)

    # The function 'split' has two forms: now use the form which splits a 
    # function in order to access its data
    u0, e0 = w0.split()
    u1, e1 = w1.split()

    ### TIMESTEPPING ###

    # Store multiple functions
    u1.rename("Fluid velocity")
    e1.rename("Free surface displacement")

    # Choose a final time and initialise arrays and files
    T = 40.0
    ufile = File('plots/variable_depth_tank_SW.pvd')
    t = 0.0
    ufile.write(u1, e1, time=t)

    # Initialise a dump counter and enter the timeloop, writing to file at
    # each dump
    ndump = 10
    dumpn = 0
    while (t < T - 0.5*dt):
        t += dt
        print "t = ", t
        # To implement the timestepping algorithm, call the solver and 
        # assign w1 to w0.
        usolver.solve()
        w0.assign(w1)
        # Dump the data
        dumpn += 1
        if dumpn == ndump:
            dumpn -= ndump
            ufile.write(u1, e1, time=t)

elif (kappa == 0):

    ### WEAK PROBLEM ###

    # Build the weak form of the timestepping algorithm, expressed as a 
    # mixed nonlinear problem
    v, xi = TestFunctions(W)

    # Here we split up a function so it can be inserted into a UFL
    # expression      
    u0, e0 = split(w0)

    # Establish the bilinear form - a function of the output function w1
    L = (
        (- Dt*inner((e0+h)*u0, grad(xi)))*dx\
        + Dt*(inner(dot(u0, nabla_grad(u0)), v)\
        + nu*inner(grad(u0), grad(v)) + g*inner(grad(e0), v))*dx
        + Dt*Cb*sqrt(dot(u0,u0))*inner(u1/(e1+h),v)*dx
    )

    # Set up the nonlinear problem
    uprob = NonlinearVariationalProblem(L, w0)
    usolver = NonlinearVariationalSolver(uprob)

    # The function 'split' has two forms: now use the form which splits a 
    # function in order to access its data
    u0, e0 = w0.split()
    
    ### PLOT SOLUTION ###

    import matplotlib.pyplot as plt
    plot(e0)
    plt.show()

else:
    raise ValueError('Enter 0 or 1!') 
