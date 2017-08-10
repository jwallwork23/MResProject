from firedrake import *
import numpy as np
from numpy import linalg as la
from scipy import linalg as sla


def construct_hessian(mesh, V, sol, method='dL2'):
    """
    Reconstructs the hessian of a scalar solution field with respect to the current mesh. The code for the integration 
    by parts reconstruction approach is based on the Monge-Amp\`ere tutorial provided in the Firedrake website 
    documentation.
    
    :param mesh: current mesh on which variables are defined.
    :param V: TensorFunctionSpace defined on ``mesh``.
    :param sol: P1 solution field defined on ``mesh``.
    :param method: mode of Hessian reconstruction; either a double L2 projection ('dL2') or as integration by parts 
    ('parts').
    :return: reconstructed Hessian associated with ``sol``.
    """

    # Construct functions:
    H = Function(V)
    tau = TestFunction(V)
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
        Lh = (inner(tau, H) + inner(div(tau), grad(sol)) ) * dx
        Lh -= (tau[0, 1] * nhat[1] * sol.dx(0) + tau[1, 0] * nhat[0] * sol.dx(1)) * ds
        Lh -= (tau[0, 0] * nhat[1] * sol.dx(0) + tau[1, 1] * nhat[0] * sol.dx(1)) * ds        # Term not in tutorial
    elif method == 'dL2':
        # Hessian reconstruction using a double L2 projection:
        V = VectorFunctionSpace(mesh, 'CG', 1)
        g = Function(V)
        psi = TestFunction(V)
        Lg = (inner(g, psi) - inner(grad(sol), psi)) * dx
        g_prob = NonlinearVariationalProblem(Lg, g)
        g_solv = NonlinearVariationalSolver(g_prob, solver_parameters=params)
        g_solv.solve()
        Lh = (inner(tau, H) + inner(div(tau), g)) * dx
        Lh -= (tau[0, 1] * nhat[1] * g[0] + tau[1, 0] * nhat[0] * g[1]) * ds
        Lh -= (tau[0, 0] * nhat[1] * g[0] + tau[1, 1] * nhat[0] * g[1]) * ds

    H_prob = NonlinearVariationalProblem(Lh, H)
    H_solv = NonlinearVariationalSolver(H_prob, solver_parameters=params)
    H_solv.solve()

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

    ia2 = 1. / (a ** 2)             # Inverse square aspect ratio
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
            lam1 = max(lam1, ia2 * lam_max)
            lam2 = max(lam2, ia2 * lam_max)

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
            lam1 = max(lam1, ia2 * lam_max)
            lam2 = max(lam2, ia2 * lam_max)

            # Reconstruct edited Hessian:
            M.dat.data[i][0, 0] = lam1 * v1[0] * v1[0] + lam2 * v2[0] * v2[0]
            M.dat.data[i][0, 1] = lam1 * v1[0] * v1[1] + lam2 * v2[0] * v2[1]
            M.dat.data[i][1, 0] = M.dat.data[i][0, 1]
            M.dat.data[i][1, 1] = lam1 * v1[1] * v1[1] + lam2 * v2[1] * v2[1]

    else:
        raise ValueError('Normalisation selection not recognised, choose `manual` or `lp`.')

    return M


def metric_gradation(mesh, V, M):

    # Specify growth parameter:
    beta = 1.4
    ln_beta = np.log(beta)

    # Get vertices and edges of mesh:
    plex = mesh._plex
    vStart, vEnd = plex.getDepthStratum(0)
    numVer = vEnd - vStart
    eStart, eEnd = plex.getDepthStratum(1)
    numEdg = eEnd - eStart
    x, y = SpatialCoordinate(mesh)

    # Establish arrays for storage:
    v12 = np.zeros(2)
    v21 = np.zeros(2)
    met1 = np.zeros((3, 3))
    met2 = np.zeros((3, 3))
    grownMet1 = np.zeros((3, 3))
    grownMet2 = np.zeros((3, 3))

    # Create a list of tags for vertices:
    verTag = np.zeros(numVer)
    for v in range(numVer):
        verTag[v] = 1
    correction = True
    i = 0

    while correction & (i < 500):
        i += 1
        correction = False

        for e in range(eStart, eEnd):
            cone = plex.getCone(e)          # Get vertices associated with edge e
            iVer1 = cone[0] - vStart
            iVer2 = cone[1] - vStart
            iMet1 = 4 * iVer1
            iMet2 = 4 * iVer2

            if (verTag[iVer1] < i) & (verTag[iVer2] < i):
                continue

            # Assemble local metrics:                   TODO: need we use this reduced form?
            met1[0] = M[iMet1]
            met1[1] = M[iMet1 + 1]
            met1[2] = M[iMet1 + 3]
            met2[0] = M[iMet2]
            met2[1] = M[iMet2 + 1]
            met2[2] = M[iMet2 + 3]

            # Calculate edge lengths and scale factor:
            v12[0] = x[iVer2] - x[iVer1]
            v12[1] = y[iVer2] - y[iVer1]
            v21[0] = - v12[0]
            v21[1] = - v12[1]
            edgLen1 = np.sqrt(np.dot(v12, np.dot(met1, v12)))
            edgLen2 = np.sqrt(np.dot(v21, np.dot(met2, v21)))
            eta2_12 = 1. / pow(1 + edgLen1 * ln_beta, 2)
            eta2_21 = 1. / pow(1 + edgLen2 * ln_beta, 2)

            for j in range(3):
                grownMet1[j] = eta2_12 * met1[j]
                grownMet2[j] = eta2_21 * met2[j]

            metNew1 = metric_intersection(mesh, V, met1, grownMet2)
            metNew2 = metric_intersection(mesh, V, met2, grownMet1)

            diff = np.abs(met1[0] - metNew1[0]) + np.abs(met1[1] - metNew1[1]) + np.abs(met1[2] - metNew1[2])
            diff /= (np.abs(met1[0]) + np.abs(met1[1]) + np.abs(met1[2]))

            if diff > 1.e-3:
                M[iMet1] = metNew1[0]
                M[iMet1 + 1] = metNew1[1]
                M[iMet1 + 2] = metNew1[1]
                M[iMet1 + 3] = metNew1[2]
                verTag[iVer1] = i + 1
                correction = True

            diff = np.abs(met2[0] - metNew2[0]) + np.abs(met2[1] - metNew2[1]) + np.abs(met2[2] - metNew2[2])
            diff /= (np.abs(met2[0]) + np.abs(met2[1]) + np.abs(met2[2]))
            if diff > 1.e-3:
                M[iMet2] = metNew2[0]
                M[iMet2+1] = metNew2[1]
                M[iMet2+2] = metNew2[1]
                M[iMet2+3] = metNew2[2]
                verTag[iVer2] = i + 1
                correction = True


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
