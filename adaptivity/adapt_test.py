from firedrake import *
import numpy as np

def adapt(mesh,metric):
    
    dim = mesh._topological_dimension
    entity_dofs = np.zeros(dim+1, dtype=np.int32)
    entity_dofs[0] = mesh.geometric_dimension()
    coordSection = mesh._plex.createSection([1], entity_dofs, perm=mesh.topology._plex_renumbering)
    
    plex = mesh._plex
    vStart, vEnd = plex.getDepthStratum(0)
    nbrVer = vEnd - vStart
#    print  "DEBUG  vStart: %d  vEnd: %d" % (vStart, vEnd)
#    coordSection.view()
    
    dmCoords = mesh.topology._plex.getCoordinateDM()
    dmCoords.setDefaultSection(coordSection)    
#    dmCoords.setDefaultSection(mesh.coordinates.function_space()._dm.getDefaultSection())

    #### TEMPORARY (?) HACK to sort the metric in the right order (waiting for Matt Knepley fix in plexadapt)
    
    met = np.ndarray(shape=metric.dat.data.shape, dtype=metric.dat.data.dtype, order='C');
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

######################################################################

## Define original mesh, with associated function spaces and functions:

mesh1 = SquareMesh(2, 2, 1, 1)
V1 = TensorFunctionSpace(mesh1, 'CG', 1) # Function space for metric
metric = Function(V1)
# TASK: generate this using a function:
metric.interpolate(Expression([[100, 0], [0, 100]]))

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

ufile = File('adapt_plots/test1.pvd')
ufile.write(f)
ufile = File('adapt_plots/test2.pvd')
ufile.write(g)


