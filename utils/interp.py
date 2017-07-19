from firedrake import *


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
            if eps >= 1e8:
                print '#### Playing with epsilons failed. Abort.'
                exit(23)

        fields_new += (f_new,)
    return fields_new


def interp_Taylor_Hood(adaptor, u_, eta_):
    """
    Transfers a mixed shallow water solution triple from the old mesh to the new mesh.

    :arg fields: tuple of functions defined on the old mesh that one wants to transfer
    """

    mesh = adaptor.adapted_mesh
    dim = mesh._topological_dimension
    assert (dim == 2)  # 3D implementation not yet considered

    W = VectorFunctionSpace(mesh, 'CG', 2) * FunctionSpace(mesh, 'CG', 1)
    q = Function(W)
    u, eta = q.split()
    notInDomain = []

    P1coords = adaptor.adapted_mesh.coordinates.dat.data
    P2coords = Function(W.sub(0)).interpolate(adaptor.adapted_mesh.coordinates).dat.data

    # Establish which vertices fall outside the domain:
    for x in range(len(P2coords)):
        try:
            valu = u_.at(P2coords[x])
        except PointNotInDomainError:
            valu = [0., 0.]
            notInDomain.append(x)
        finally:
            u.dat.data[x] = valu

    eps = 1e-6  # For playing with epsilons
    while len(notInDomain) > 0:
        print '#### Number of points not in domain for velocity: %d / %d' % (len(notInDomain), len(P2coords))
        eps *= 10
        print '#### Trying epsilon = ', eps
        for x in notInDomain:
            try:
                valu = u_.at(P2coords[x], tolerance=eps)
            except PointNotInDomainError:
                valu = [0., 0.]
            finally:
                u.dat.data[x] = valu
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
        if eps >= 1e8:
            print '#### Playing with epsilons failed. Abort.'
            exit(23)

    return u, eta, q, W