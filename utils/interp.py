from firedrake import *

from adaptivity import *

class Meshd:
    """
    A structure holding the objects related to a mesh, courtesy of Nicolas Barral.
    """

    def __init__(self, mesh, reorderPlex = True, computeAltMin = True):

        self.mesh = mesh

        self.V = FunctionSpace(self.mesh, 'CG', 1)

        self.altMin = Function(self.V)

        entity_dofs = np.zeros(self.mesh._topological_dimension + 1, dtype=np.int32)
        entity_dofs[0] = self.mesh.geometric_dimension()
        self.section = self.mesh._plex.createSection([1], entity_dofs, perm=self.mesh.topology._plex_renumbering)

        if reorderPlex :
            with self.mesh.coordinates.dat.vec_ro as coords:
                self.mesh.topology._plex.setCoordinatesLocal(coords)

        if computeAltMin :
            if self.mesh._topological_dimension == 2:
                self.altMin.interpolate(2 * CellVolume(self.mesh) / MaxCellEdgeLength(self.mesh))
            else :
                print '#### 3D implementation not yet considered.'


def interp(adaptor, *fields, **kwargs):
    """
    Transfers a solution field from the old mesh to the new mesh.

    :arg fields: tuple of functions defined on the old mesh that one wants to transfer
    """

    fields_new = ()
    for f in fields:
        V_new = functionspace.FunctionSpace(adaptor.adapted_mesh, f.function_space().ufl_element())
        f_new = function.Function(V_new)
        notInDomain = []

        if f.ufl_element().family() == 'Lagrange' and f.ufl_element().degree() == 1:
            coords = adaptor.adapted_mesh.coordinates.dat.data
        elif f.ufl_element().family() == 'Lagrange':
            degree = f.ufl_element().degree()
            C = functionspace.VectorFunctionSpace(adaptor.adapted_mesh, 'CG', degree)
            interp_coordinates = function.Function(C)
            interp_coordinates.interpolate(adaptor.adapted_mesh.coordinates)
            coords = interp_coordinates.dat.data
        else:
            raise NotImplementedError("Can only interpolate CG fields")

        try:
            f_new.dat.data[:] = f.at(coords)
        except PointNotInDomainError:
            print '#### Points not in domain! Time to play with epsilons'

            mesh = adaptor.adapted_mesh
            dim = mesh._topological_dimension
            assert (dim == 2)                           # 3D implementation not yet considered
            meshd = Meshd(mesh)
            plexnew = mesh._plex
            vStart, vEnd = plexnew.getDepthStratum(0)   # Vertices of new plex

            # Establish which vertices fall outside the domain:
            for v in range(vStart, vEnd):
                offnew = meshd.section.getOffset(v) / dim
                newCrd = mesh.coordinates.dat.data[offnew]
                try:
                    val = f.at(newCrd)
                except PointNotInDomainError:
                    val = 0.
                    notInDomain.append(v)
                finally:
                    f_new.dat.data[offnew] = val

        eps = 1e-6  # For playing with epsilons
        while len(notInDomain) > 0:
            print '#### Number of points not in domain: %d / %d' % \
                  (len(notInDomain), mesh.topology.num_vertices())
            eps *= 10
            print '#### Trying epsilon = ', eps
            for v in notInDomain:
                offnew = meshd.section.getOffset(v) / dim
                newCrd = mesh.coordinates.dat.data[offnew]
                try:
                    val = f.at(newCrd, tolerance=eps)
                except PointNotInDomainError:
                    val = 0.
                finally:
                    f_new.dat.data[offnew] = val
                    notInDomain.remove(v)
            if eps >= 100:
                print '#### Playing with epsilons failed. Abort.'
                exit(23)

        fields_new += (f_new,)
    return fields_new

def relab(*fields):
    return fields