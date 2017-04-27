from firedrake import *
import numpy as np

################################################################################################################################

def adapt(mesh, metric):
    '''A function which generates a new mesh, provided with a previous mesh and an adaptivity metric. Courtesy of Nicolas
    Barral.'''
    
    dim = mesh._topological_dimension
    entity_dofs = np.zeros(dim+1, dtype=np.int32)
    entity_dofs[0] = mesh.geometric_dimension()
    coordSection = mesh._plex.createSection([1], entity_dofs, perm=mesh.topology._plex_renumbering)
    
    plex = mesh._plex
    vStart, vEnd = plex.getDepthStratum(0)
    nbrVer = vEnd - vStart
    
    dmCoords = mesh.topology._plex.getCoordinateDM()
    dmCoords.setDefaultSection(coordSection)    
    
    met = np.ndarray(shape=metric.dat.data.shape, dtype=metric.dat.data.dtype, order='C');
    for iVer in range(nbrVer):
        off = coordSection.getOffset(iVer+vStart)/dim

        met[iVer] = metric.dat.data[off]
    for iVer in range(nbrVer):
        metric.dat.data[iVer] = met[iVer]

    with mesh.coordinates.dat.vec_ro as coords:
        mesh.topology._plex.setCoordinatesLocal(coords)
    with metric.dat.vec_ro as vec:
        newplex = dmplex.petscAdap(mesh.topology._plex, vec)

    newmesh = Mesh(newplex)

    return newmesh


def construct_hessian(mesh, sol):
    '''A function which computes the hessian of a scalar solution field with respect to the current mesh.'''

    # Construct functions:
    H = Function(Vm)            # Hessian-to-be
    sigma = TestFunction(Vm)
    nhat = FacetNormal(mesh)    # Normal vector

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

    return H


def compute_steady_metric(mesh, H, sol, h_min = 0.005, h_max = 0.3, a = 100):
    '''A function which computes the steady metric for remeshing, provided with the current mesh, hessian and free surface.
    Here h_min and h_max denote the respective minimum and maxiumum tolerated side-lengths, while a denotes the maximum
    tolerated aspect ratio. This code is based on Nicolas Barral's function ``computeSteadyMetric``, from ``adapt.py``.'''

    sol_min = 0.01
    ia = 1./(a**2)
    ihmin2 = 1./(h_min**2)
    ihmax2 = 1./(h_max**2)
    M = H
    
    for i in range(mesh.topology.num_vertices()):
        
        # Generate local Hessian, scaling to avoid roundoff error and editing values:
        H_loc = H.dat.data[i] * 1/max(abs(sol.dat.data[i]), sol_min) * n2
## TODO:                         what is this mysterious scale factor? ^^
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

################################################################################################################################

## Define original mesh, with associated function spaces and functions:

mesh1 = SquareMesh(2, 2, 1, 1)
V1 = TensorFunctionSpace(mesh1, 'CG', 1) # Function space for metric
metric = Function(V1)

# TASK: generate the metric using a function involving the Hessian:
metric.interpolate(Expression([['10*x[0]', 0], [0, '10*x[1]']]))
# UNDERSTAND: what exactly the role of the Hessian is, and how it
#           : affects the adaption.

F = FunctionSpace(mesh1, 'CG', 1)        # Scalar function space
f = Function(F)
f.interpolate(Expression('x[0]'))

## Adapt mesh and set up new function spaces:

mesh2 = adapt(mesh1, metric)
V2 = TensorFunctionSpace(mesh2, 'CG', 1)
metric2 = Function(V2)
G = FunctionSpace(mesh2, 'CG', 1)

## Interpolate functions onto new mesh:

g = Function(G) # NOTE: .at doesn't like points outside the domain
g.dat.data[:] = f.at(mesh2.coordinates.dat.data)

## Plot results:

File('adapt_plots/sensor_test1.pvd').write(f)
File('adapt_plots/sensor_test2.pvd').write(g)
