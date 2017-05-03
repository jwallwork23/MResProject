from firedrake import *
import numpy as np
from numpy import linalg as LA

from interp import *

def adapt(mesh, metric):
    '''A function which generates a new mesh, provided with a previous mesh and an adaptivity metric. Courtesy of Nicolas
    Barral.'''
    
    dim = mesh._topological_dimension
    entity_dofs = np.zeros(dim+1, dtype=np.int32)
    entity_dofs[0] = mesh.geometric_dimension()
    coordSection = mesh._plex.createSection(\
        [1], entity_dofs, perm=mesh.topology._plex_renumbering)
    
    plex = mesh._plex
    vStart, vEnd = plex.getDepthStratum(0)
    nbrVer = vEnd - vStart
##    print  "DEBUG  vStart: %d  vEnd: %d" % (vStart, vEnd)
##    coordSection.view()
    
    dmCoords = mesh.topology._plex.getCoordinateDM()
    dmCoords.setDefaultSection(coordSection)    
##    dmCoords.setDefaultSection(\
##        mesh.coordinates.function_space()._dm.getDefaultSection())

    #### TEMPORARY (?) HACK to sort the metric in the right order
    ####                (waiting for Matt Knepley fix in plexadapt)
    
    met = np.ndarray(shape=metric.dat.data.shape, \
                     dtype=metric.dat.data.dtype, order='C');
    for iVer in range(nbrVer):
        off = coordSection.getOffset(iVer+vStart)/dim
#        print "DEBUG  iVer: %d  off: %d   nbrVer: %d" %(iVer, off, nbrVer)
        met[iVer] = metric.dat.data[off]
    for iVer in range(nbrVer):
        metric.dat.data[iVer] = met[iVer]
#    metric.dat.data.data = met.data

    with mesh.coordinates.dat.vec_ro as coords:
        mesh.topology._plex.setCoordinatesLocal(coords)
    with metric.dat.vec_ro as vec:
        newplex = dmplex.petscAdap(mesh.topology._plex, vec)

    newmesh = Mesh(newplex)

    return newmesh

def construct_hessian(mesh, sol):
    '''A function which computes the hessian of a scalar solution field with respect to the current mesh.'''

    # Construct function spaces and functions:
    V = TensorFunctionSpace(mesh, 'CG', 1)
    H = Function(V)                             # Hessian-to-be
    sigma = TestFunction(V)
    nhat = FacetNormal(mesh)                    # Normal vector

    # Establish and solve a variational problem associated with the Monge-Ampere equation:
    Lh = (
            inner(H, sigma) * dx + inner(div(sigma), grad(sol)) * dx - \
            (sigma[0,1] * nhat[1] * sol.dx(0) + sigma[1,0] * nhat[0] * sol.dx(1)) * ds
        )
    H_prob = NonlinearVariationalProblem(Lh, H)
    H_solv = NonlinearVariationalSolver(H_prob, solver_parameters={'snes_rtol': 1e8,
                                                                   'ksp_rtol': 1e-5,
                                                                   'ksp_gmres_restart': 20,
                                                                   'pc_type': 'sor',
                                                                   'snes_monitor': True,
                                                                   'snes_view': False,
                                                                   'ksp_monitor_true_residual': False,
                                                                   'snes_converged_reason': True,
                                                                   'ksp_converged_reason': True,})
    H_solv.solve()

    return H, V

def compute_steady_metric(mesh, V, H, sol, epsilon = 0.01, h_min = 0.005, h_max = 0.1, a = 100.):
    '''A function which computes the steady metric for remeshing, provided with the current mesh, hessian and free surface.
    Here h_min and h_max denote the respective minimum and maxiumum tolerated side-lengths, while a denotes the maximum
    tolerated aspect ratio. This code is based on Nicolas Barral's function ``computeSteadyMetric``, from ``adapt.py``.'''

    # Set maximum and minimum parameters:
    sol_min = 0.01
    ia = 1./(a**2)
    ihmin2 = 1./(h_min**2)
    ihmax2 = 1./(h_max**2)

    # Establish metric object:
    M = Function(V)
    M = H
    
    for i in range(mesh.topology.num_vertices()):
        
        # Generate local Hessian, scaling to avoid roundoff error and editing values:
        H_loc = H.dat.data[i] * 1/(max(abs(sol.dat.data[i]), sol_min) * epsilon)
        mean_diag = 0.5 * (H_loc[0][1] + H_loc[1][0])
        H_loc[0][1] = mean_diag; H_loc[1][0] = mean_diag

        # Find eigenpairs and truncate eigenvalues:
        lam, v = LA.eig(H_loc)
        v1, v2 = v[0], v[1]
        lam1 = min(ihmin2, max(ihmax2, abs(lam[0])))
        lam2 = min(ihmin2, max(ihmax2, abs(lam[1])))
        lam_max = max(lam1, lam2)
        lam1 = max(lam1, ia * lam_max)
        lam2 = max(lam2, ia * lam_max)

        # Reconstruct edited Hessian:
        M.dat.data[i][0,0] = lam1 * v1[0] * v1[0] + lam2 * v2[0] * v2[0]
        M.dat.data[i][0,1] = lam1 * v1[0] * v1[1] + lam2 * v2[0] * v2[1]
        M.dat.data[i][1,0] = M.dat.data[i][0,1]
        M.dat.data[i][1,1] = lam1 * v1[1] * v1[1] + lam2 * v2[1] * v2[1]

    return M

def update_SW_FE(mesh1, mesh2, u_, u, eta_, eta, b):
    '''A function which updates shallow water solution fields and bathymetry from one mesh to another.'''
    
    # Establish function spaces on the new mesh:
    Vm = TensorFunctionSpace(mesh2, 'CG', 1)
    Vu = VectorFunctionSpace(mesh2, 'CG', 1)
    Ve = FunctionSpace(mesh2, 'CG', 1)
    Vq = MixedFunctionSpace((Vu, Ve))

    # Establish functions in the new spaces:
    q_2 = Function(Vq); u_2, eta_2 = q_2.split()
    q2 = Function(Vq); u2, eta2 = q2.split()
    b2 = Function(Ve)

    # Interpolate functions across from the previous mesh:
    interp(u_, mesh1, u_2, mesh2)
    interp(u, mesh1, u2, mesh2)
    interp(eta_, mesh1, eta_2, mesh2)
    interp(eta, mesh1, eta2, mesh2)
    interp(b, mesh1, b2, mesh2)

    return q_2, q2, u_2, u2, eta_2, eta2, b2, Vq
