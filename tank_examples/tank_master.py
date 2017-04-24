from firedrake import *
from thetis import *
import numpy as np
import matplotlib.pyplot as plt

######################### USEFUL FUNCTIONS ############################

def zero_boundary_nonlinear_form():
    '''Weak residual form of the nonlinear shallow water equations'''
    L = (
     (ze * (eta-eta_) - Dt * inner((eta + b) * u, grad(ze)) + \
      inner(u-u_, v) + Dt * inner(dot(u, nabla_grad(u)), v) + \
      nu * inner(grad(u), grad(v)) + Dt * g * inner(grad(eta), v) + \
      Dt * Cb * sqrt(dot(u_, u_)) * inner(u/(eta+b), v)) * dx(degree=4)   
     )
    return L

def zero_boundary_linear_form():
    '''Weak residual form of the linear shallow water equations'''
    L = (
    (ze * (eta-eta_) - Dt * inner((eta + b) * u, grad(ze)) + \
    inner(u-u_, v) + Dt * g *(inner(grad(eta), v))) * dx
    )
    return L

def nonzero_boundary_nonlinear_form():
    '''Weak residual form of the nonlinear shallow water equations'''
    # Define the outward pointing normal to the mesh
    n = FacetNormal(mesh)
    # Integrate terms of the momentum equation over the interior:
    Lu_int = (inner(u-u_, v) + Dt * (inner(dot(u, nabla_grad(u)), v)
        + nu * inner(grad(u), grad(v)) + g * inner(grad(eta), v))
        + Dt * Cb * sqrt(dot(u_, u_)) * inner(u / (eta + b), v)) * \
        dx(degree=4)
    # Integrate terms of the continuity equation over the interior:
    Le_int = (ze * (eta-eta_) - \
              Dt * inner((eta + b) * u, grad(ze))) * dx(degree=4)
    # Integrate over left-hand boundary:
    L_side1 = Dt * (-inner(dot(n, nabla_grad(u)), v)
        + dot(u, n) * (ze * (eta + b))) * ds(1)(degree=4)
    # Integrate over right-hand boundary:
    L_side2 = Dt * (-inner(dot(n, nabla_grad(u)), v)
        + dot(u, n) * (ze * (eta + b))) * ds(2)(degree=4)
    # Establish the bilinear form using the above integrals:
    return Lu_int + Le_int + L_side1 + L_side2

def nonzero_boundary_linear_form():
    '''Weak residual form of the linear shallow water equations'''
    # Define the outward pointing normal to the mesh
    n = FacetNormal(mesh)
    # Integrate terms of the momentum equation over the interior:
    Lu_int = (inner(u-u_, v) + Dt * g *(inner(grad(eta), v))) * dx
    # Integrate terms of the continuity equation over the interior:
    Le_int = (ze * (eta-eta_) - \
              Dt * inner((eta + b) * u, grad(ze))) * dx
    # Integrate over left-hand boundary:
    L_side1 = dot(u, n) * (ze * (eta + b)) * ds(1)
    # Integrate over right-hand boundary:
    L_side2 = dot(u, n) * (ze * (eta + b)) * ds(2)
    # Establish the bilinear form using the above integrals:
    return Lu_int + Le_int + L_side1 + L_side2

# For imposing time-dependent BCs:
def wave_machine(t, A, p, in_flux):
    '''Time-dependent flux function'''
    return A * sin(2 * pi * t / p) + in_flux

########################### PARAMETERS ################################

# Specify problem parameters:
compare = raw_input('Use standalone, Thetis or both? (s/t/b): ') or 's'
if ((compare != 's') & (compare != 't') & (compare != 'b')):
    raise ValueError('Please try again, choosing s, t or b.')
if (compare != 't'):
    mode = raw_input('Use linear or nonlinear equations? (l/n): ') \
           or 'l'
    if ((mode != 'l') & (mode != 'n')):
        raise ValueError('Please try again, choosing l or n.')
else:
    mode = 't'
bath = raw_input('Consider non-trivial bathymetry? (y/n): ') or 'n'
if ((bath != 'y') & (bath != 'n')):
    raise ValueError('Please try again, choosing y or n.')
waves = raw_input('Consider a wave generator? (y/n): ') or 'n'
if ((waves != 'y') & (waves != 'n')):
    raise ValueError('Please try again, choosing y or n.')
n = int(raw_input('Specify number of cells per m (default 30): ') \
        or 30)
dt = float(raw_input('Specify timestep (default 0.01): ') or 0.01)
Dt = Constant(dt)   # TODO: adaptive timestepping?
ndump = 10
t_export = dt*ndump
T = float(raw_input('Specify simulation duration in s (default 40): ') \
    or 40.)

# Set physical parameters for the scheme:
nu = 1e-3           # Viscosity (kg s^{-1} m^{-1})
g = 9.81            # Gravitational acceleration (m s^{-2})
Cb = 0.0025         # Bottom friction coefficient (dimensionless)
depth = 0.1         # Specify tank water depth (m)

# 'Wave generator' parameters
A = 0.01            # 'Tide' amplitude (m)
p = 0.5             # 'Tide' period (s)
in_flux = 0         # Flux into domain

############################ FE SETUP #################################

# Define domain and mesh:
lx = 4; ly = 1; nx = lx*n; ny = ly*n
mesh = RectangleMesh(nx, ny, lx, ly)
x = SpatialCoordinate(mesh)

# Define function spaces:
Vu  = VectorFunctionSpace(mesh, 'CG', 2)    # \ Use Taylor-Hood
Ve = FunctionSpace(mesh, 'CG', 1)           # / elements
Vq = MixedFunctionSpace((Vu, Ve))

# Construct a function to store our two variables at time n:
q_ = Function(Vq)       
u_, eta_ = q_.split()

# Interpolate bathymetry:
b = Function(Ve, name = 'Bathymetry')
if (bath == 'n'):
    # Construct a (constant) bathymetry function:
    b.assign(depth)
elif (bath == 'y'):
    b.interpolate(0.1 + 0.04 * sin(2*pi*x[0]) * sin(2*pi*x[1]))
    File('screenshots/tank_bathymetry.pvd').write(b)

################## INITIAL AND BOUNDARY CONDITIONS ####################

# Interpolate ICs:
u_.interpolate(Expression([0, 0]))
if (waves == 'n'):
    eta_.interpolate(-0.01*cos(0.5*pi*x[0]))
else:
    eta_.interpolate(Expression(0))
    # Establish a BC object for the oscillating inflow condition
    bcval = Constant(0.0)
    bc1 = DirichletBC(Vq.sub(1), bcval, 1)
    # Apply no-slip BC to eta on the right end of the domain
    bc2 = DirichletBC(Vq.sub(1), (0.0), 2)

########################## WEAK PROBLEM ###############################

# Build the weak form of the timestepping algorithm, expressed as a 
# mixed nonlinear problem
v, ze = TestFunctions(Vq)
q = Function(Vq)
q.assign(q_)
u, eta = split(q)
u_, eta_ = split(q_)

# Establish form:
if (mode == 'l'):
    if (waves == 'n'):
        L = zero_boundary_linear_form()
    else:
        L = nonzero_boundary_linear_form()
else:
    if (waves == 'n'):
        L = zero_boundary_nonlinear_form()
    else:
        L = nonzero_boundary_nonlinear_form()

# Set up the variational problem:
if (waves == 'n'):
    q_prob = NonlinearVariationalProblem(L, q)
    q_solve = NonlinearVariationalSolver(q_prob,
        solver_parameters={
                            'mat_type': 'matfree',
                            'snes_type': 'ksponly',
                            'pc_type': 'python',
                            'pc_python_type': 'firedrake.AssembledPC',
                            'assembled_pc_type': 'lu',
                            'snes_lag_preconditioner': -1, 
                            'snes_lag_preconditioner_persists': True,
                            })
else:
    q_prob = NonlinearVariationalProblem(L, q, bcs=[bc1, bc2])
    q_solve = NonlinearVariationalSolver(q_prob,
        solver_parameters={
                            'ksp_type': 'gmres',
                            'ksp_rtol': '1e-8',
                            'pc_type': 'fieldsplit',
                            'pc_fieldsplit_type': 'schur',
                            'pc_fieldsplit_schur_fact_type': 'full',
                            'fieldsplit_0_ksp_type': 'cg',
                            'fieldsplit_0_pc_type': 'lu',
                            'fieldsplit_1_ksp_type': 'cg',
                            'fieldsplit_1_pc_type': 'hypre',
                            'pc_fieldsplit_schur_precondition': 'selfp',
                            })

# Split functions in order to access their data:
u_, eta_ = q_.split()
u, eta = q.split()

# Store multiple functions
u.rename('Fluid velocity')
eta.rename('Free surface displacement')

############################ TIMESTEPPING #############################

# Initialise output directory:
if (bath == 'n'):
    if (waves == 'n'):
        if (mode == 'l'):
            q_file = File('tank_outputs/model_prob1_linear.pvd')
        else:
            q_file = File('tank_outputs/model_prob1_nonlinear.pvd')
    else:
        if (mode == 'l'):
            q_file = File('tank_outputs/model_prob2_linear.pvd')
        else:
            q_file = File('tank_outputs/model_prob2_nonlinear.pvd')
elif (bath == 'y'):
    if (waves == 'n'):
        if (mode == 'l'):
            q_file = File('tank_outputs/model_prob3_linear.pvd')
        else:
            q_file = File('tank_outputs/model_prob3_nonlinear.pvd')
    else:
        if (mode == 'l'):
            q_file = File('tank_outputs/model_prob4_linear.pvd')
        else:
            q_file = File('tank_outputs/model_prob4_nonlinear.pvd')

# Initialise time, arrays and dump counter:
t = 0.0; dumpn = 0; i = 0
q_file.write(u, eta, time=t)

# Initialise arrays for storage:
eta_vals = np.zeros((int(T/(ndump*dt))+1, (nx+1)*(ny+1)))
u_vals = np.zeros((int(T/(ndump*dt))+1, (2*nx+1)*(2*ny+1), 2))
eta_vals[i,:] = eta.dat.data
u_vals[i,:,:] = u.dat.data

if (compare != 't'):
    # Enter the timeloop:
    while (t < T - 0.5*dt):     
        t += dt
        if (waves == 'y'):
            bcval.assign(wave_machine(t, A, p, in_flux))    # Update BC
        q_solve.solve()
        q_.assign(q)
        dumpn += 1
        # Dump data:
        if (dumpn == ndump):
            print 't = ', t, ' seconds'
            dumpn -= ndump
            i += 1
            q_file.write(u, eta, time=t)
            eta_vals[i,:] = eta.dat.data
            u_vals[i,:,:] = u.dat.data

############################ THETIS SETUP #############################

if (compare != 's'):
    # Construct solver:
    solver_obj = solver2d.FlowSolver2d(mesh, b)
    options = solver_obj.options
    options.t_export = t_export
    options.t_end = T
    options.outputdir = 'tank_outputs'

    # Specify integrator of choice:
    options.timestepper_type = 'backwardeuler'  # Implicit timestepping
    options.dt = dt

    # Specify initial surface elevation:
    elev_init = Function(Ve, name = 'Initial elevation')
    x = SpatialCoordinate(mesh)
    if (waves == 'n'):
        elev_init.interpolate(-0.01*cos(0.5*pi*x[0]))
        solver_obj.assign_initial_conditions(elev=elev_init)
        
    else:
        # Define boundary IDs of the domain, for convenience:
        left_bnd_id = 1
        right_bnd_id = 2

        # Specify BCs as a dictionary:
        swe_bnd = {}
        in_flux = 0
        swe_bnd[right_bnd_id] = {'elev': Constant(0.0), \
                                 'flux': Constant(-in_flux)}
        # NOTE: -ve value => flow into domain. ( Defined as outward
        # normal flux)

        # Initialise BCs:
        tide_flux_const = Constant(wave_machine(0, A, p, in_flux))
        swe_bnd[left_bnd_id] = {'flux': tide_flux_const}

        # Assign BCs to solver object
        solver_obj.bnd_functions['shallow_water'] = swe_bnd
        # NOTE: If BCs are not assigned for some boundaries (the
        # lateral boundaries 3 and 4 in this case), Thetis assumes
        # impermeable land conditions.
        
        # Re-evaluate the BCs as the simulation progresses
        def update_forcings(t_new):
            """Callback function that updates all time dependent
            forcing fields"""
            tide_flux_const.assign(wave_machine(t_new, A, p, in_flux))

    if (compare == 'b'):

        # Re-initialise counters and set up error arrays:
        dumpn = 0; i = 0
        u_err = np.zeros((int(T/(ndump*dt))+1))
        eta_err = np.zeros((int(T/(ndump*dt))+1)) 

        # Initialise Taylor-Hood versions of eta and u:
        eta_t = Function(Ve)
        u_t = Function(Vu)

        def plot_error():
            '''A function which approximates the error made by the
            standalone solver, as compared against Thetis' solution.
            '''
            global i
            # Interpolate functions onto the same spaces:
            eta_t.interpolate( \
                solver_obj.fields.solution_2d.split()[1])
            u_t.interpolate( \
                solver_obj.fields.solution_2d.split()[0])
            eta.dat.data[:] = eta_vals[i,:]
            u.dat.data[:] = u_vals[i,:,:]
            # Calculate (absolute) errors:
            u_err[i] = errornorm(u, u_t)
            eta_err[i] = errornorm(eta, eta_t)
            i += 1
    
        # Run the model:
        if (waves == 'y'):
            solver_obj.iterate(update_forcings=update_forcings, \
                           export_func=plot_error)
        else:
            solver_obj.iterate(export_func=plot_error)
    else:
        if (waves == 'y'):
            solver_obj.iterate(update_forcings=update_forcings)
        else:
            solver_obj.iterate()
    
############################# PLOT ERROR #############################

if (compare == 'b'):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.plot(np.linspace(0, T, int(T/(ndump*dt))+1), u_err,
             label='Fluid velocity error')
    plt.plot(np.linspace(0, T, int(T/(ndump*dt))+1), eta_err,
             label='Free surface error')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               borderaxespad=0.)
    plt.xlim([0, 7200])
    plt.xlabel(r'Time (s)')
    plt.ylabel(r'L2 error')
    plt.savefig('tank_outputs/graphs/error_{y1}_{y2}_{y3}.png'\
                    .format(y1=mode, y2=bath, y3=waves))
