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
    mesh = adaptor.adapted_mesh
    dim = mesh._topological_dimension
    assert (dim == 2)  # 3D implementation not yet considered

    fields_new = ()
    for f in fields:
        V_new = FunctionSpace(adaptor.adapted_mesh, f.function_space().ufl_element())
        f_new = Function(V_new)
        notInDomain = []

        if f.ufl_element().family() == 'Lagrange' and f.ufl_element().degree() == 1:
            coords = adaptor.adapted_mesh.coordinates.dat.data                      # Vertex/node coords
        elif f.ufl_element().family() == 'Lagrange':
            degree = f.ufl_element().degree()
            C = VectorFunctionSpace(adaptor.adapted_mesh, 'CG', degree)
            interp_coordinates = Function(C)
            interp_coordinates.interpolate(adaptor.adapted_mesh.coordinates)
            coords = interp_coordinates.dat.data                                    # Node coords (NOT just vertices)
        else:
            raise NotImplementedError("Can only interpolate CG fields")

        try:
            f_new.dat.data[:] = f.at(coords)
        except PointNotInDomainError:
            print '#### Points not in domain! Commence attempts by increasing tolerances'

            # Establish which vertices fall outside the domain:
            for x in range(len(coords)):
                try:
                    val = f.at(coords[x])
                except PointNotInDomainError:
                    val = 0.
                    notInDomain.append(x)
                finally:
                    f_new.dat.data[x] = val
        eps = 1e-6                              # Tolerance to be increased
        while len(notInDomain) > 0:
            print '#### Number of points not in domain: %d / %d' % (len(notInDomain), len(coords))
            eps *= 10
            print '#### Trying tolerance = ', eps
            for x in notInDomain:
                try:
                    val = f.at(coords[x], tolerance=eps)
                except PointNotInDomainError:
                    val = 0.
                finally:
                    f_new.dat.data[x] = val
                    notInDomain.remove(x)
            if eps >= 1e5:
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

    # Establish functions in the new spaces:
    q_new = Function(W)
    u_new, v_new, eta_new = q_new.split()
    qnew = Function(W)
    unew, vnew, etanew = qnew.split()

    # Interpolate functions across from the previous mesh:
    u_new, unew, v_new, vnew, eta_new, etanew = interp(adaptor, u_, u, v_, v, eta_, eta)

    return q_new, qnew, u_new, unew, v_new, vnew, eta_new, etanew, W


def interp_mixed(adaptor, u_, v_, eta_):
    """
    Transfers a mixed shallow water solution triple from the old mesh to the new mesh.

    :arg fields: tuple of functions defined on the old mesh that one wants to transfer
    """

    mesh = adaptor.adapted_mesh
    dim = mesh._topological_dimension
    assert (dim == 2)  # 3D implementation not yet considered

    W = FunctionSpace(mesh, 'CG', 2) * FunctionSpace(mesh, 'CG', 2) * FunctionSpace(mesh, 'CG', 1)
    q = Function(W)
    u, v, eta = q.split()
    notInDomain = []

    P1coords = adaptor.adapted_mesh.coordinates.dat.data
    interp_coordinates = Function(VectorFunctionSpace(adaptor.adapted_mesh, 'CG', 2))
    interp_coordinates.interpolate(adaptor.adapted_mesh.coordinates)
    P2coords = interp_coordinates.dat.data

    try:
        u.dat.data[:] = u_.at(P2coords)
        v.dat.data[:] = v_.at(P2coords)
        eta.dat.data[:] = eta_.at(P1coords)
    except PointNotInDomainError:
        print '#### Points not in domain! Commence attempts by increasing tolerances'

        # Establish which vertices fall outside the domain:
        for x in range(len(P2coords)):
            try:
                valu = u_.at(P2coords[x])
                valv = v_.at(P2coords[x])
            except PointNotInDomainError:
                valu = 0.
                valv = 0.
                notInDomain.append(x)
            finally:
                u.dat.data[x] = valu
                v.dat.data[x] = valv

        eps = 1e-6  # For playing with epsilons
        while len(notInDomain) > 0:
            print '#### Number of points not in domain for velocity: %d / %d' % (len(notInDomain), len(P2coords))
            eps *= 10
            print '#### Trying epsilon = ', eps
            for x in notInDomain:
                try:
                    valu = u_.at(P2coords[x], tolerance=eps)
                    valv = v_.at(P2coords[x], tolerance=eps)
                except PointNotInDomainError:
                    valu = 0.
                    valv = 0.
                finally:
                    u.dat.data[x] = valu
                    v.dat.data[x] = valv
                    notInDomain.remove(x)
            if eps >= 1e5:
                print '#### Playing with epsilons failed. Abort.'
                exit(23)
        assert (len(notInDomain) == 0)  # All nodes should have been brought back into the domain

        # Establish which vertices fall outside the domain:
        for x in range(len(P1coords)):
            try:
                val = eta_.at(P1coords[x])
            except PointNotInDomainError:
                val = 0.
                notInDomain.append(x)
            finally:
                eta.dat.data[x] = val

        eps = 1e-6  # For playing with epsilons
        while len(notInDomain) > 0:
            print '#### Number of points not in domain for free surface: %d / %d' % (len(notInDomain), len(P1coords))
            eps *= 10
            print '#### Trying epsilon = ', eps
            for x in notInDomain:
                try:
                    val = eta_.at(P1coords[x], tolerance=eps)
                except PointNotInDomainError:
                    val = 0.
                finally:
                    eta.dat.data[x] = val
                    notInDomain.remove(x)
            if eps >= 1e5:
                print '#### Playing with epsilons failed. Abort.'
                exit(23)

    return u, v, eta, q, W