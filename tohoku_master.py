from firedrake import *
from thetis import *

from math import radians
import numpy as np
import matplotlib.pyplot as plt
from time import clock

from domain import *
from forms import *

######################################################## PARAMETERS ###########################################################

# Specify solver parameters:
compare = raw_input('Use standalone, Thetis or both? (s/t/b): ') or 's'
if ((compare != 's') & (compare != 't') & (compare != 'b')):
    raise ValueError('Please try again, choosing s, t or b.')
if (compare != 't'):
    mode = raw_input('Use linear or nonlinear equations? (l/n): ') or 'l'
    if ((mode != 'l') & (mode != 'n')):
        raise ValueError('Please try again, choosing l or n.')
else:
    mode = 't'
res = raw_input('Mesh type fine, medium or coarse? (f/m/c): ') or 'c'
if ((res != 'f') & (res != 'm') & (res != 'c')):
    raise ValueError('Please try again, choosing f, m or c.')
dt = float(raw_input('Specify timestep (s) (default 15): ') or 15.)
Dt = Constant(dt)
ndump = 4           # Timesteps per data dump
t_export = ndump * dt
T = float(raw_input('Specify time period (s) (default 7200): ') or 7200.)
tmode = raw_input('Time-averaging mode? (y/n, default n): ') or 'n'
if ((tmode != 'y') & (tmode != 'n')):
    raise ValueError('Please try again, choosing y or n.')

########################################################### SETUP #############################################################

mesh, Vq, q_, u_, eta_, b = Tohoku_domain(res)

##################################################### FORWARD WEAK PROBLEM ####################################################

if (compare != 't'):
    # Build the weak form of the timestepping algorithm, expressed as a mixed nonlinear problem:
    v, ze = TestFunctions(Vq)
    q = Function(Vq)
    q.assign(q_)
    u, eta = split(q)
    u_, eta_ = split(q_)

    # Establish form:
    if (mode == 'l'):
        L1 = linear_form(u, u_, eta, eta_, v, ze, b, Dt)
    elif (mode == 'n'):
        L1 = nonlinear_form(u, u_, eta, eta_, v, ze, b, Dt)

    # Set up the variational problem:
    params = {'ksp_type': 'gmres', 'ksp_rtol': '1e-8',
              'pc_type': 'fieldsplit', 'pc_fieldsplit_type': 'schur',
              'pc_fieldsplit_schur_fact_type': 'full',
              'fieldsplit_0_ksp_type': 'cg', 'fieldsplit_0_pc_type': 'ilu',
              'fieldsplit_1_ksp_type': 'cg', 'fieldsplit_1_pc_type': 'hypre',
              'pc_fieldsplit_schur_precondition': 'selfp',}
    q_prob = NonlinearVariationalProblem(L1, q)
    q_solve = NonlinearVariationalSolver(q_prob, solver_parameters=params)

    # Split functions in order to access their data:
    u_, eta_ = q_.split()
    u, eta = q.split()

    # Store multiple functions:
    u.rename('Fluid velocity'); eta.rename('Free surface displacement')

    ################################################ FORWARD TIMESTEPPING #####################################################

    # Initialise output directory and dump counter:
    if (mode == 'l'):
        q_file = File('plots/tsunami_outputs/tohoku_linear.pvd')
    elif (mode == 'n'):
        q_file = File('plots/tsunami_outputs/tohoku_nonlinear.pvd')
    t = 0.0; i = 0; dumpn = 0
    q_file.write(u, eta, time=t)

    # Initialise arrays for storage:
    eta_vals = np.zeros((int(T/(ndump*dt))+1, 1099))    # \ TODO: Make  
    u_vals = np.zeros((int(T/(ndump*dt))+1, 4067, 2))   # \ more general to
    eta_vals[i,:] = eta.dat.data                        #   apply in fine
    u_vals[i,:,:] = u.dat.data                          #   and med cases

    tic1 = clock()
    # Enter the timeloop:
    while (t < T - 0.5*dt):     
        t += dt
        q_solve.solve()
        q_.assign(q)
        dumpn += 1
        # Dump data:
        if (dumpn == ndump):
            print 't = ', t/60, ' mins'
            dumpn -= ndump
            i += 1
            q_file.write(u, eta, time=t)
            if (tmode == 'n'):
                eta_vals[i,:] = eta.dat.data
                u_vals[i,:,:] = u.dat.data
    toc1 = clock()

    print 'Elapsed time for standalone solver: %1.2es' % (toc1 - tic1)
                  
## TODO: Implement damage measures

################################################# FORWARD THETIS SETUP ########################################################

if (compare != 's'):
    # Construct solver:
    solver_obj = solver2d.FlowSolver2d(mesh, b)
    options = solver_obj.options
    options.t_export = t_export
    options.t_end = T
    options.outputdir = 'tsunami_outputs'

    # Specify integrator of choice:
    options.timestepper_type = 'backwardeuler'  # Implicit timestepping
    options.dt = dt

    # Specify initial surface elevation:
    solver_obj.assign_initial_conditions(elev=eta0)

    if (compare == 'b'):

        # Re-initialise counters and set up error arrays:
        i = 0
        u_err = np.zeros((int(T/(ndump*dt))+1))
        eta_err = np.zeros((int(T/(ndump*dt))+1)) 

        # Initialise Taylor-Hood versions of eta and u:
        eta_t = Function(Ve)
        u_t = Function(Vu)

        def compute_error():
            '''A function which approximates the error made by the
            standalone solver, as compared against Thetis' solution.
            '''
            global i
            # Interpolate functions onto the same spaces:
            eta_t.interpolate(solver_obj.fields.solution_2d.split()[1])
            u_t.interpolate(solver_obj.fields.solution_2d.split()[0])
            eta.dat.data[:] = eta_vals[i,:]
            u.dat.data[:] = u_vals[i,:,:]
            # Calculate (relative) errors:
            if ((norm(u_t) > 1e-4) & (norm(eta_t) > 1e-4)):
                u_err[i] = errornorm(u, u_t)/norm(u_t)
                eta_err[i] = errornorm(eta, eta_t)/norm(eta_t)
            else:
                u_err[i] = errornorm(u, u_t)
                eta_err[i] = errornorm(eta, eta_t)
            i += 1

    # Run solver:
        if (tmode == 'y'):
            tic2 = clock()
            solver_obj.iterate(export_func=compute_error)
            toc2 = clock()
            print 'Elapsed time for Thetis solver: %1.2es' % (toc2 - tic2)
        else:
            solver_obj.iterate()
        
    else:
        solver_obj.iterate()

######################################################### PLOT ERROR ##########################################################

if (compare == 'b'):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.plot(np.linspace(0, T, int(T/(ndump*dt))+1), u_err, label='Fluid velocity error')
    plt.plot(np.linspace(0, T, int(T/(ndump*dt))+1), eta_err, label='Free surface error')
    plt.legend(bbox_to_anchor=(0.1, 1.02, 1., .102), loc=3, borderaxespad=0.)
    plt.xlim([0, 7200])
    plt.xlabel(r'Time (s)')
    plt.ylabel(r'Relative L2 error')
    plt.savefig('plots/tsunami_outputs/screenshots/error_{y1}_{y2}.png'.format(y1=mode, y2=res))

################################################## ADJOINT 'INITIAL' CONDITIONS ###############################################

# TODO: Specify some ICs

##################################################### ADJOINT WEAK PROBLEM ####################################################

# Establish test functions and split adjoint variables:
w, xi = TestFunctions(Vq)
lam = Function(Vq)
lam.assign(lam_)
lm, le = split(lam)      
lm_, le_ = split(lam_)

# Establish form:
if (mode == 'l'):
    L2 = adj_linear_form()
elif (mode == 'n'):
    L2 = adj_nonlinear_form()

# Set up the variational problem
lam_prob = NonlinearVariationalProblem(L2, lam)
lam_solve = NonlinearVariationalSolver(lam_prob, solver_parameters=params)

# Split functions in order to access their data:
lm_, le_ = lam_.split(); lm, le = lam.split()

# Store multiple functions:
lm.rename('Adjoint fluid momentum'); le.rename('Adjoint free surface displacement')

##################################################### BACKWARD TIMESTEPPING ###################################################

# Initialise some arrays for storage:
le_vals = np.zeros((int(T/(ndump*dt))+1, 1099))
lm_vals = np.zeros((int(T/(ndump*dt))+1, 4067, 2))
i -= 1
le_vals[i,:] = le.dat.data;
lm_vals[i,:] = lm.dat.data

# Initialise dump counter and files:
if (dumpn == 0):
    dumpn = ndump
if (mode == 'l'):
    lam_file = File('plots/tsunami_outputs/tohoku_linear_adj.pvd')
else:
    lam_file = File('plots/tsunami_outputs/tohoku_nonlinear_adj.pvd')
lam_file.write(lm, le, time=0)

# Enter the backward timeloop:
while (t > 0):
    t -= dt
    print 't = ', t, ' seconds'
    lam_solve.solve()
    lam_.assign(lam)
    dumpn -= 1
    # Dump data:
    if (dumpn == 0):
        dumpn += ndump
        i -= 1
        lm_vals[i,:] = lm.dat.data
        le_vals[i,:] = le.dat.data
        # Note the time inversion in output:
        lam_file.write(lm, le, time=T-t)

#################################################### ADJOINT THETIS SETUP #####################################################

# TODO : use Firedrake adjoint?
