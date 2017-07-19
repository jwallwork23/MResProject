from firedrake import *
from adaptivity import *


class Meshd:
    """
    A structure holding the objects related to a mesh, courtesy of Nicolas Barral.
    """

    def __init__(self, mesh, reorderPlex=True, computeAltMin=True):

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
    mesh = adaptor.adapted_mesh
    dim = mesh._topological_dimension
    assert (dim == 2)                   # 3D implementation not yet considered

    for f in fields:
        V_new = FunctionSpace(mesh, f.function_space().ufl_element())
        f_new = Function(V_new)
        notInDomain = []

        if f.ufl_element().family() == 'Lagrange' and f.ufl_element().degree() == 1:
            mesh_int = mesh
        elif f.ufl_element().family() == 'Lagrange':
            degree = f.ufl_element().degree()
            C = VectorFunctionSpace(mesh, 'CG', degree)
            interp_coordinates = Function(C)
            interp_coordinates.interpolate(mesh.coordinates)
            mesh_int = interp_coordinates.function_space().mesh()           # Mesh of interpolated nodes
        else:
            raise NotImplementedError("Can only interpolate CG fields")
        coords = mesh_int.coordinates.dat.data

        try:
            f_new.dat.data[:] = f.at(coords)
        except PointNotInDomainError:
            print '#### Points not in domain! Time to play with epsilons'

            meshd = Meshd(mesh_int)
            plex = mesh_int._plex
            vStart, vEnd = plex.getDepthStratum(0)                  # Node list

            # Establish which vertices fall outside the domain:
            for v in range(vStart, vEnd):
                off = meshd.section.getOffset(v) / dim              # Node number
                newCrd = mesh_int.coordinates.dat.data[off]         # Coord thereof
                try:
                    val = f.at(newCrd)
                except PointNotInDomainError:
                    val = 0.
                    notInDomain.append(v)
                finally:
                    f_new.dat.data[off] = val

        eps = 1e-6  # For playing with epsilons
        while len(notInDomain) > 0:
            print '#### Number of points not in domain: %d / %d' % (len(notInDomain), mesh.topology.num_vertices())
            eps *= 10
            print '#### Trying epsilon = ', eps
            for v in notInDomain:
                off = meshd.section.getOffset(v) / dim
                newCrd = mesh_int.coordinates.dat.data[off]
                try:
                    val = f.at(newCrd, tolerance=eps)
                except PointNotInDomainError:
                    val = 0.
                finally:
                    f_new.dat.data[off] = val
                    notInDomain.remove(v)
            if eps >= 100:
                print '#### Playing with epsilons failed. Abort.'
                exit(23)

        fields_new += (f_new,)
    return fields_new

def update_SW(adaptor, u_, u, eta_, eta):
    """A function which updates shallow water solution fields and bathymetry from one mesh to another."""

    # Get mesh and establish a mixed function space thereupon:
    mesh = adaptor.adapted_mesh
    W = MixedFunctionSpace((VectorFunctionSpace(mesh, 'CG', 2), FunctionSpace(mesh, 'CG', 1)))

    # Establish functions in the new spaces:
    q_new = Function(W)
    u_new, eta_new = q_new.split()
    qnew = Function(W)
    unew, etanew = qnew.split()

    # Interpolate functions across from the previous mesh:
    u_new, unew, eta_new, etanew = interp(adaptor, u_, u, eta_, eta)

    return q_new, qnew, u_new, unew, eta_new, etanew, W

def update_SW2(adaptor, u_, u, v_, v, eta_, eta):
    """A function which updates shallow water solution fields and bathymetry from one mesh to another."""

    # Get mesh and establish a mixed function space thereupon:
    mesh = adaptor.adapted_mesh
    W = FunctionSpace(mesh, 'CG', 2) * FunctionSpace(mesh, 'CG', 2) * FunctionSpace(mesh, 'CG', 1)

    # Interpolate functions across from the previous mesh:
    u_new, unew, v_new, vnew, eta_new, etanew = interp(adaptor, u_, u, v_, v, eta_, eta)

    # Establish functions in the new spaces:
    q_new = Function(W)
    u_new, v_new, eta_new = split(q_new)
    qnew = Function(W)
    unew, vnew, etanew = split(qnew)

    return q_new, qnew, u_new, unew, v_new, vnew, eta_new, etanew, W


def interp_mixed(adaptor, u_, v_, eta_):
    """
    Transfers a solution field from the old mesh to the new mesh.

    :arg fields: tuple of functions defined on the old mesh that one wants to transfer
    """

    W = FunctionSpace(mesh, 'CG', 2) * FunctionSpace(mesh, 'CG', 2) * FunctionSpace(mesh, 'CG', 1)
    q = Function(W)
    u, v, eta = q.split()
    notInDomain = []

    P1coords = adaptor.adapted_mesh.coordinates.dat.data
    C = VectorFunctionSpace(adaptor.adapted_mesh, 'CG', 2)
    Function(C).interpolate(adaptor.adapted_mesh.coordinates)
    P2coords = interp_coordinates.dat.data

    try:
        u.dat.data[:] = u_.at(P2coords)
        v.dat.data[:] = v_.at(P2coords)
        eta.dat.data[:] = eta_.at(P1coords)
    except PointNotInDomainError:
        print '#### Points not in domain! Time to play with epsilons'

        mesh = adaptor.adapted_mesh
        mesh_ = P
        dim = mesh._topological_dimension
        assert (dim == 2)                           # 3D implementation not yet considered
        meshd = Meshd(mesh)
        plex = mesh._plex
        vStart, vEnd = plex.getDepthStratum(0)   # Vertices of new plex

        # Establish which vertices fall outside the domain:
        for ver in range(vStart, vEnd):
            offnew = meshd.section.getOffset(ver) / dim
            newCrd = mesh.coordinates.dat.data[offnew]
            try:
                valu = u_.at(newCrd)
                valv = v_.at(newCrd)
                vale = eta_.at(newCrd)
            except PointNotInDomainError:
                valu = 0.
                valv = 0.
                vale = 0.
                notInDomain.append(ver)
            finally:
                u.dat.data[offnew] = valu
                v.dat.data[offnew] = valv
                eta.dat.data[offnew] = vale

        eps = 1e-6  # For playing with epsilons
        while len(notInDomain) > 0:
            print '#### Number of points not in domain: %d / %d' % (len(notInDomain), mesh.topology.num_vertices())
            eps *= 10
            print '#### Trying epsilon = ', eps
            for ver in notInDomain:
                offnew = meshd.section.getOffset(ver) / dim
                newCrd = mesh.coordinates.dat.data[offnew]
                try:
                    valu = u_.at(newCrd, tolerance=eps)
                    valv = v_.at(newCrd, tolerance=eps)
                    vale = eta_.at(newCrd, tolerance=eps)
                except PointNotInDomainError:
                    valu = 0.
                    valv = 0.
                    vale = 0.
                finally:
                    u.dat.data[offnew] = valu
                    v.dat.data[offnew] = valv
                    e.dat.data[offnew] = vale
                    notInDomain.remove(ver)
            if eps >= 100:
                print '#### Playing with epsilons failed. Abort.'
                exit(23)

    return u, v, eta