from firedrake import *
import numpy as np
from numpy import linalg as la
from scipy import linalg as sla

from interp import *

def adapt(mesh, metric) :
    """A function which generates a new mesh, provided with a previous mesh and an adaptivity metric. Courtesy of 
    Nicolas Barral."""

    # Establish topological and geometric dimensions:
    dim = mesh._topological_dimension
    entity_dofs = np.zeros(dim + 1, dtype = np.int32)
    entity_dofs[0] = mesh.geometric_dimension()

    # Generate list of dimensions and offsets vertices and faces (for visualisation, use .view):
    coordSection = mesh._plex.createSection([1], entity_dofs, perm = mesh.topology._plex_renumbering)

    # Get the DMPlex object encapsulating the mesh topology and determine the vertices of plex to consider:
    plex = mesh._plex
    vStart, vEnd = plex.getDepthStratum(0)
    nbrVer = vEnd - vStart

    # Establish DM coordinates (a DM is an abstract PETSc object that manages an abstract grid object and its
    # interactions with the algebraic solvers):
    dmCoords = mesh.topology._plex.getCoordinateDM()
    dmCoords.setDefaultSection(coordSection)    
##    dmCoords.setDefaultSection(mesh.coordinates.function_space()._dm.getDefaultSection())

#### TEMPORARY (?) HACK to sort the metric in the right order (waiting for Matt Knepley fix in plexadapt)

    # Establish a new metric as a numpy array ('C' denoting column style):
    met = np.ndarray(shape = metric.dat.data.shape, dtype = metric.dat.data.dtype, order = 'C')

    # Loop over vertices of the mesh (?):
    for iVer in range(nbrVer) :

        # Establish offsets of vertices (?):
        off = coordSection.getOffset(iVer + vStart) / dim
##        print "DEBUG  iVer: %d  off: %d   nbrVer: %d" %(iVer, off, nbrVer)

        # Transfer offsets into new metric:
        met[iVer] = metric.dat.data[off]

    # Overwrite metric with new metric (could use metric.dat.data.data = met.data):
    for iVer in range(nbrVer) :
        metric.dat.data[iVer] = met[iVer]

    # Construct new mesh from metric:
    with mesh.coordinates.dat.vec_ro as coords :
        mesh.topology._plex.setCoordinatesLocal(coords)
    with metric.dat.vec_ro as vec :
        newplex = dmplex.petscAdap(mesh.topology._plex, vec)
    newmesh = Mesh(newplex)

    return newmesh

def construct_hessian(mesh, V, sol) :
    """A function which computes the hessian of a scalar solution field with respect to the current mesh."""

    # Construct functions:
    H = Function(V)                             # Hessian-to-be
    sigma = TestFunction(V)
    nhat = FacetNormal(mesh)                    # Normal vector

    # Establish and solve a variational problem associated with the Monge-Ampere equation:
    Lh = (
            (inner(sigma, H) + inner(div(sigma), grad(sol)) ) * dx -
            (sigma[0,1] * nhat[1] * sol.dx(0) + sigma[1,0] * nhat[0] * sol.dx(1)) * ds
        )
    H_prob = NonlinearVariationalProblem(Lh, H)
    H_solv = NonlinearVariationalSolver(H_prob, solver_parameters = {'snes_rtol' : 1e8,
                                                                     'ksp_rtol' : 1e-5,
                                                                     'ksp_gmres_restart' : 20,
                                                                     'pc_type' : 'sor',
                                                                     'snes_monitor' : False,
                                                                     'snes_view' : False,
                                                                     'ksp_monitor_true_residual' : False,
                                                                     'snes_converged_reason' : False,
                                                                     'ksp_converged_reason' : False, })
    H_solv.solve()

    return H

def compute_steady_metric(mesh, V, H, sol, h_min = 0.005, h_max = 0.1, a = 100., normalise = 'lp', p = 2, N = 1000.,
                          ieps = 1000.) :
    """A function which computes the steady metric for re-meshing, provided with the current mesh, hessian and 
    free surface. Here h_min and h_max denote the respective minimum and maximum tolerated side-lengths, while a 
    denotes the maximum tolerated aspect ratio. Further,  N denotes the target number of nodes and ieps denotes the
    inverse of the target error for the two respective normalisation approaches. This code is based on Nicolas Barral's 
    function ``computeSteadyMetric``, from ``adapt.py``."""

    # Set parameter values:
    ia = 1. / (a ** 2)
    ihmin2 = 1. / (h_min ** 2)
    ihmax2 = 1. / (h_max ** 2)

    # Establish metric object:
    M = Function(V)

    if normalise == 'manual' :
    
        for i in range(mesh.topology.num_vertices()) :

            # Specify minimum tolerated value for the solution field:
            sol_min = 0.001
        
            # Generate local Hessian:
            H_loc = H.dat.data[i] * ieps / (max(abs(sol.dat.data[i]), sol_min))  # To avoid round-off error
            mean_diag = 0.5 * (H_loc[0][1] + H_loc[1][0])
            H_loc[0][1] = mean_diag
            H_loc[1][0] = mean_diag

            # Find eigenpairs and truncate eigenvalues:
            lam, v = la.eig(H_loc)
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

    elif normalise == 'lp' :

        # Establish determinant object:
        detH = Function(FunctionSpace(mesh, 'CG', 1))

        for i in range(mesh.topology.num_vertices()) :

            # Generate local Hessian:
            H_loc = H.dat.data[i]
            mean_diag = 0.5 * (H_loc[0][1] + H_loc[1][0])
            H_loc[0][1] = mean_diag
            H_loc[1][0] = mean_diag

            # Find eigenpairs of Hessian and truncate eigenvalues:
            lam, v = la.eig(H_loc)
            v1, v2 = v[0], v[1]
            lam1 = max(abs(lam[0]), 1e-10)                                      # \ To avoid round-off error
            lam2 = max(abs(lam[1]), 1e-10)                                      # /
            det = lam1 * lam2

            # Reconstruct edited Hessian and rescale:
            M.dat.data[i][0,0] = lam1 * v1[0] * v1[0] + lam2 * v2[0] * v2[0]
            M.dat.data[i][0,1] = lam1 * v1[0] * v1[1] + lam2 * v2[0] * v2[1]
            M.dat.data[i][1,0] = M.dat.data[i][0,1]
            M.dat.data[i][1,1] = lam1 * v1[1] * v1[1] + lam2 * v2[1] * v2[1]
            M.dat.data[i] *= pow(det, -1. / (2 * p + 2))
            detH.dat.data[i] = pow(det, p / (2. * p + 2))

        detH_integral = assemble(detH * dx)
        M *= N / detH_integral                                                  # Scale by the target number of vertices

        for i in range(mesh.topology.num_vertices()) :

            # Find eigenpairs of metric and truncate eigenvalues:
            lam, v = la.eig(M.dat.data[i])
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

    else :
        raise ValueError('Normalisation selection not recognised, choose `manual` or `lp`.')

    return M

def metric_intersection(mesh, V, M1, M2) :
    """A function which computes the metric with respect to two different fields."""

    # Establish metric intersection object:
    M12 = Function(V)

    for i in range(mesh.topology.num_vertices()) :

        M = M1.dat.data[i]
        iM = la.inv(M)
        Mbar = np.transpose(sla.sqrtm(iM)) * M2.dat.data[i] * sla.sqrtm(iM)
        lam, v = la.eig(Mbar)
        M12.dat.data[i] = v * [[max(lam[0], 1), 0], [0, max(lam[1], 1)]] * np.transpose(v)
        M12.dat.data[i] = np.transpose(sla.sqrtm(M)) * M12.dat.data[i] * sla.sqrtm(M)

    return M12

def update_SW(meshd1, meshd2, u_, u, eta_, eta) :
    """A function which updates shallow water solution fields and bathymetry from one mesh to another."""

    # Get mesh:
    mesh2 = meshd2.mesh

    # Establish function spaces on the new mesh:
    Vu = VectorFunctionSpace(mesh2, 'CG', 1)        # TODO: use Taylor-Hood
    Ve = FunctionSpace(mesh2, 'CG', 1)
    Vq = MixedFunctionSpace((Vu, Ve))

    # Establish functions in the new spaces:
    q_2 = Function(Vq)
    u_2, eta_2 = q_2.split()
    q2 = Function(Vq)
    u2, eta2 = q2.split()

    # Interpolate functions across from the previous mesh:
    interp(u_, meshd1, u_2, meshd2)
    interp(u, meshd1, u2, meshd2)
    interp(eta_, meshd1, eta_2, meshd2)
    interp(eta, meshd1, eta2, meshd2)

    return q_2, q2, u_2, u2, eta_2, eta2, Vq

def update_variable(meshd1, meshd2, phi) :
    """Interpolate a variable from one mesh onto another."""

    phi2 = Function(meshd2.V)
    interp(phi, meshd1, phi2, meshd2)

    return phi2