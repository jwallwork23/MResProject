from firedrake import *
import numpy as np
from numpy import linalg as la
from scipy import linalg as sla


def construct_hessian(mesh, V, sol, method='dL2', treat_boundaries='off'):
    """
    Reconstructs the hessian of a scalar solution field with respect to the current mesh. The code for the integration 
    by parts reconstruction approach is based on the Monge-Amp\`ere tutorial provided in the Firedrake website 
    documentation.
    
    :param mesh: current mesh on which variables are defined.
    :param V: TensorFunctionSpace defined on ``mesh``.
    :param sol: P1 solution field defined on ``mesh``.
    :param method: mode of Hessian reconstruction; either a double L2 projection ('dL2') or as integration by parts 
    ('parts').
    :param treat_boundaries: specify whether or not to interpolate boundary values using a finite difference stencil. 
    :return: reconstructed Hessian associated with ``sol``.
    """

    # Construct functions:
    H = Function(V)
    sigma = TestFunction(V)
    nhat = FacetNormal(mesh)                    # Normal vector
    params = {'snes_rtol': 1e8,
              'ksp_rtol': 1e-5,
              'ksp_gmres_restart': 20,
              'pc_type': 'sor',
              'snes_monitor': False,
              'snes_view': False,
              'ksp_monitor_true_residual': False,
              'snes_converged_reason': False,
              'ksp_converged_reason': False, }

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
    H_solv = NonlinearVariationalSolver(H_prob, solver_parameters=params)
    H_solv.solve()

    # if treat_boundaries == 'on':
        # assert mesh._topological_dimension == 2                                     # 3D not yet considered
        # plex = mesh._plex
        # vStart, vEnd = plex.getDepthStratum(0)                                      # Vertices

        # [Get b_nodes and i_nodes]

        # noIntNbrs = []

        # [Create some functions int_nbrs and bdy_nbrs]

        # for v in b_nodes:

        #     if len(int_nbrs(v)) == 0:
        #         noIntNbrs.append(v)
        #     else :
        #         num = np.zeros((2, 2))
        #         den = 0
        #         for w in int_nbrs(v) :
        #             [Calculate num += H.dat.data[w] * Mass(w) and den += Mass(w)]
        #         H.dat.data[v] = num/den

        # if len(noIntNbrs) > 0 :
        #     for v in noIntNbrs :
        #         num = np.zeros((2, 2))
        #         den = 0
        #         for w in bdy_nbrs(v) :
        #             [Calculate num += H.dat.data[w] * Mass(w) and den += Mass(w)]
        #         H.dat.data[v] = num/den

    # TODO: Complete finite difference approximation at boundaries. Perhaps use plex.getAdjacency?

    return H


def compute_steady_metric(mesh, V, H, sol, h_min=0.005, h_max=0.1, a=100., normalise='lp', p=2, num=1000., ieps=1000.):
    """
    Computes the steady metric for mesh adaptation. Based on Nicolas Barral's function ``computeSteadyMetric``, from 
    ``adapt.py``, 2016.
    
    :param mesh: current mesh on which variables are defined.
    :param V: TensorFunctionSpace defined on ``mesh``.
    :param H: reconstructed Hessian, usually chosen to be associated with ``sol``.
    :param sol: P1 solution field defined on ``mesh``.
    :param h_min: minimum tolerated side-lengths.
    :param h_max: maximum tolerated side-lengths.
    :param a: maximum tolerated aspect ratio.
    :param normalise: mode of normalisation; either a manual rescaling ('manual') or an Lp approach ('Lp').
    :param p: norm order in the Lp normalisation approach, where ``p => 1`` and ``p = infty`` is an option.
    :param num: target number of nodes, in the case of Lp normalisation.
    :param ieps: inverse of the target error, in the case of manual normalisation.
    :return: steady metric associated with Hessian H.
    """

    ia = 1. / (a ** 2)              # Inverse square aspect ratio
    ihmin2 = 1. / (h_min ** 2)      # Inverse square minimal side-length
    ihmax2 = 1. / (h_max ** 2)      # Inverse square maximal side-length
    M = Function(V)

    if normalise == 'manual':
        for i in range(mesh.topology.num_vertices()):
            sol_min = 0.001     # Minimum tolerated value for the solution field
        
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
            M.dat.data[i][0, 0] = lam1 * v1[0] * v1[0] + lam2 * v2[0] * v2[0]
            M.dat.data[i][0, 1] = lam1 * v1[0] * v1[1] + lam2 * v2[0] * v2[1]
            M.dat.data[i][1, 0] = M.dat.data[i][0, 1]
            M.dat.data[i][1, 1] = lam1 * v1[1] * v1[1] + lam2 * v2[1] * v2[1]

    elif normalise == 'lp':
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
            M.dat.data[i][0, 0] = lam1 * v1[0] * v1[0] + lam2 * v2[0] * v2[0]
            M.dat.data[i][0, 1] = lam1 * v1[0] * v1[1] + lam2 * v2[0] * v2[1]
            M.dat.data[i][1, 0] = M.dat.data[i][0, 1]
            M.dat.data[i][1, 1] = lam1 * v1[1] * v1[1] + lam2 * v2[1] * v2[1]
            M.dat.data[i] *= pow(det, -1. / (2 * p + 2))
            detH.dat.data[i] = pow(det, p / (2. * p + 2))

        detH_integral = assemble(detH * dx)
        M *= num / detH_integral                                                # Scale by the target number of vertices
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
            M.dat.data[i][0, 0] = lam1 * v1[0] * v1[0] + lam2 * v2[0] * v2[0]
            M.dat.data[i][0, 1] = lam1 * v1[0] * v1[1] + lam2 * v2[0] * v2[1]
            M.dat.data[i][1, 0] = M.dat.data[i][0, 1]
            M.dat.data[i][1, 1] = lam1 * v1[1] * v1[1] + lam2 * v2[1] * v2[1]

    else:
        raise ValueError('Normalisation selection not recognised, choose `manual` or `lp`.')

    return M


def metric_intersection(mesh, V, M1, M2):
    """
    Intersect two metric fields.
    
    :param mesh: current mesh on which variables are defined.
    :param V: TensorFunctionSpace defined on ``mesh``.
    :param M1: first metric to be intersected.
    :param M2: second metric to be intersected.
    :return: intersection of metrics M1 and M2.
    """
    M12 = Function(V)
    for i in range(mesh.topology.num_vertices()):
        M = M1.dat.data[i]
        iM = la.inv(M)
        Mbar = np.transpose(sla.sqrtm(iM)) * M2.dat.data[i] * sla.sqrtm(iM)
        lam, v = la.eig(Mbar)
        M12.dat.data[i] = v * [[max(lam[0], 1), 0], [0, max(lam[1], 1)]] * np.transpose(v)
        M12.dat.data[i] = np.transpose(sla.sqrtm(M)) * M12.dat.data[i] * sla.sqrtm(M)

    return M12
