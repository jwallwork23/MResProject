from firedrake import *
import numpy as np
from numpy import linalg as la
from scipy import linalg as sla

from interp import *

def construct_hessian(mesh, V, sol, method='parts'):
    """
    A function which computes the hessian of a scalar solution field with respect to the current mesh. This code is
    based on that provided in the Monge-Ampere tutorial provided on the Firedrake website.
    """

    # Construct functions:
    H = Function(V)                             # Hessian-to-be
    sigma = TestFunction(V)
    nhat = FacetNormal(mesh)                    # Normal vector
    params = {'snes_rtol' : 1e8,
              'ksp_rtol' : 1e-5,
              'ksp_gmres_restart' : 20,
              'pc_type' : 'sor',
              'snes_monitor' : False,
              'snes_view' : False,
              'ksp_monitor_true_residual' : False,
              'snes_converged_reason' : False,
              'ksp_converged_reason' : False, }

    if method == 'parts':
        # Hessian reconstruction using integration by parts:
        Lh = (inner(sigma, H) + inner(div(sigma), grad(sol)) ) * dx
        Lh -= (sigma[0,1] * nhat[1] * sol.dx(0) + sigma[1,0] * nhat[0] * sol.dx(1)) * ds
        Lh -= (sigma[0,0] * nhat[1] * sol.dx(0) + sigma[1,1] * nhat[0] * sol.dx(1)) * ds        # Term not in tutorial
    elif method == 'dL2':
        # Hessian reconstruction using a double L2 projection:
        W = VectorFunctionSpace(mesh, 'CG', 1)
        phi = Function(W)
        psi = TestFunction(W)
        Lg = (inner(phi, psi) - inner(psi, grad(sol))) * dx
        g_prob = NonlinearVariationalProblem(Lg, phi)
        g_solv = NonlinearVariationalSolver(g_prob, solver_parameters=params)
        g_solv.solve()
        Lh = (inner(sigma, H) + inner(div(sigma), phi)) * dx
        Lh -= (sigma[0, 1] * nhat[1] * phi[0] + sigma[1, 0] * nhat[0] * phi[1]) * ds
        Lh -= (sigma[0, 0] * nhat[1] * phi[0] + sigma[1, 1] * nhat[0] * phi[1]) * ds

    H_prob = NonlinearVariationalProblem(Lh, H)
    H_solv = NonlinearVariationalSolver(H_prob, solver_parameters = params)
    H_solv.solve()

    return H

# TODO: implement also a double L2 projection

# def construct_hessian2(meshd, V, sol) :
#      """
#      A function which computes the Hessian of a scalar solution field w.r.t. the current mesh.
#      """
#
#     mesh = meshd.mesh
#     dim = mesh._topological_dimension
#     assert (dim == 2)                           # 3D implementation not yet considered
#
#     # Get the DMPlex object encapsulating the mesh topology and determine the vertices of plex to consider:
#     plex = mesh._plex
#     vStart, vEnd = plex.getDepthStratum(0)      # Vertices of new plex
#     bc = DirichletBC(V, 0, on_boundary)
#     b_nodes = bc.nodes
#
#     params = {'snes_rtol' : 1e8,
#               'ksp_rtol' : 1e-5,
#               'ksp_gmres_restart' : 20,
#               'pc_type' : 'sor',
#               'snes_monitor' : False,
#               'snes_view' : False,
#               'ksp_monitor_true_residual' : False,
#               'snes_converged_reason' : False,
#               'ksp_converged_reason' : False, }
#
#     H = Function(V)                             # Hessian-to-be
#     sigma = TestFunction(V)
#
#     for v in range(vStart, vEnd) and not b_nodes :
#
#         for i in range(2) :
#
#             for j in range(i) :
#
#                 # Compute Hij
#                 # Something like H * test * dx = 0.5 * ( sol.dx(i) + sol.dx(j)) * test * ds - 0.5 * (sol.dx(i) * test.dx(j) + sol.dx(j) * test.dx(i)) * dx
#
#         # [Put H21 = H12]
#
#     noIntNbrs = []
#
#     # [Create some functions int_nbrs and bdy_nbrs]
#
#     for v in b_nodes:
#
#         if len(int_nbrs(v)) == 0:
#             noIntNbrs.append(v)
#         else :
#             Nn = np.zeros((2, 2))
#             Dn = np.zeros((2, 2))
#             for w in int_nbrs(v) :
#                 # [Calculate Nn += H*Mnk and Dn += Mnk]
#         # [Set H = Nn/Dn]
#
#     if len(noIntNbrs) > 0 :
#         for v in noIntNbrs :
#             Nn = np.zeros((2, 2))
#             Dn = np.zeros((2, 2))
#             for w in bdy_nbrs(v) :
#                 # [Calculate Nn += H*Mnk and Dn += Mnk]
#         # [Set H = Nn/Dn]
#
#     return H

def compute_steady_metric(mesh, V, H, sol, h_min = 0.005, h_max = 0.1, a = 100., normalise = 'lp', p = 2, N = 1000.,
                          ieps = 1000.):
    """
    A function which computes the steady metric for re-meshing, provided with the current mesh, hessian and free 
    surface. Here h_min and h_max denote the respective minimum and maximum tolerated side-lengths, while a denotes the 
    maximum tolerated aspect ratio. Further,  N denotes the target number of nodes and ieps denotes the inverse of the 
    target error for the two respective normalisation approaches. This code is based on Nicolas Barral's function 
    ``computeSteadyMetric``, from ``adapt.py``.
    """

    # Set parameter values:
    ia = 1. / (a ** 2)
    ihmin2 = 1. / (h_min ** 2)
    ihmax2 = 1. / (h_max ** 2)

    # Establish metric object:
    M = Function(V)

    if normalise == 'manual':
    
        for i in range(mesh.topology.num_vertices()):

            # Specify minimum tolerated value for the solution field:
            sol_min = 0.001
        
            # Generate local Hessian:
            H_loc = H.dat.data[i] * ieps / (max(np.sqrt(assemble(sol * sol * dx)), sol_min))  # To avoid round-off error
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

    elif normalise == 'lp':

        # Establish determinant object:
        detH = Function(FunctionSpace(mesh, 'CG', 1))

        for i in range(mesh.topology.num_vertices()):

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

        for i in range(mesh.topology.num_vertices()):

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

class Meshd :
    """
    A structure holding the objects related to a mesh, courtesy of Nicolas Barral.
    """

    def __init__(self, mesh, reorderPlex = True, computeAltMin = True):

        self.mesh = mesh

        self.V = FunctionSpace(self.mesh, 'CG', 1)

        self.altMin = Function(self.V)

        entity_dofs = np.zeros(self.mesh._topological_dimension + 1, dtype = np.int32)
        entity_dofs[0] = self.mesh.geometric_dimension()
        self.section = self.mesh._plex.createSection([1], entity_dofs, perm = self.mesh.topology._plex_renumbering)

        if reorderPlex :
            with self.mesh.coordinates.dat.vec_ro as coords:
                self.mesh.topology._plex.setCoordinatesLocal(coords)

        if computeAltMin :
            if self.mesh._topological_dimension == 2:
                self.altMin.interpolate(2 * CellVolume(self.mesh) / MaxCellEdgeLength(self.mesh))
            else :
                print '#### 3D implementation not yet considered.'

def metric_intersection(mesh, V, M1, M2):
    """
    A function which computes the metric with respect to two different fields.
    """

    # Establish metric intersection object:
    M12 = Function(V)

    for i in range(mesh.topology.num_vertices()):

        M = M1.dat.data[i]
        iM = la.inv(M)
        Mbar = np.transpose(sla.sqrtm(iM)) * M2.dat.data[i] * sla.sqrtm(iM)
        lam, v = la.eig(Mbar)
        M12.dat.data[i] = v * [[max(lam[0], 1), 0], [0, max(lam[1], 1)]] * np.transpose(v)
        M12.dat.data[i] = np.transpose(sla.sqrtm(M)) * M12.dat.data[i] * sla.sqrtm(M)

    return M12