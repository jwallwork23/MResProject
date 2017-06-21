from firedrake import *
import numpy as np

INF = float("inf")


class Meshd :
    """A structure holding the objects related to a mesh."""

    def __init__(self, mesh, reorderPlex = True, computeAltMin = True) :

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
            if self.mesh._topological_dimension == 2 :
                self.altMin.interpolate(2 * CellVolume(self.mesh) / MaxCellEdgeLength(self.mesh))
            else :
                print '#### 3D implementation not yet considered.'

def barCoord(crdM, crdTri, i):
    """A function which computes the barycentric coordinate of M in triangle Tri = P0, P1, P2 with respect to the ith
    vertex crd = det(MPj, MPk) / det(PiPj, PiPk)"""

    # Get other indices using a consistent numbering order:
    j = (i + 1) % 3
    k = (i + 2) % 3

    res1 = (crdTri[j][0] - crdM[0]) * (crdTri[k][1] - crdM[1]) - (crdTri[k][0] - crdM[0]) * (crdTri[j][1] - crdM[1])
    res2 = (crdTri[j][0] - crdTri[i][0]) * (crdTri[k][1] - crdTri[i][1]) - \
           (crdTri[k][0] - crdTri[i][0]) * (crdTri[j][1] - crdTri[i][1])
    res = res1 / res2

    return res

def interp(u, meshd, unew, meshdnew) :
    """A function which interpolates a function u onto a new mesh. Only slightly modified version of Nicolas Barral's
    function ``interpol``, from the Python script ``interpol.py``."""

    # Get meshes from structures:
    mesh = meshd.mesh
    meshnew = meshdnew.mesh

    # Establish topological dimension:
    dim = mesh._topological_dimension
    assert(dim == 2)                                        # 3D implementation not yet considered

    # Get the DMPlex object encapsulating the mesh topology and determine the vertices of plex to consider:
    plexnew = meshnew._plex
    vStart, vEnd = plexnew.getDepthStratum(0)               # Vertices of new plex

    notInDomain = []

    # Establish which vertices fall outside the domain:
    for v in range(vStart, vEnd) :

        offnew = meshdnew.section.getOffset(v) / dim
        newCrd = meshnew.coordinates.dat.data[offnew]

        # Attempt to interpolate:
        try :
            val = u.at(newCrd)
        except PointNotInDomainError :
            print '####  New vertex not in domain: %f %f' % (newCrd[0], newCrd[1])
            val = 0.
            notInDomain.append([v, INF, -1])   # (Vertex, distance, edge) tuple... TODO: store newCrd here instead of v
        finally :
            unew.dat.data[offnew] = val
    
    # If there are points which fall outside the domain,
    if len(notInDomain) > 0 :
        print '####  Warning: number of points not in domain: %d / %d' % \
              (len(notInDomain), meshnew.topology.num_vertices())

        plex = mesh._plex
        fStart, fEnd = plex.getHeightStratum(1)                         # Edges/facets of old plex
        vStart, vEnd = plex.getDepthStratum(0)                          # Vertices of old plex

        # Loop over edges:
        for f in range(fStart, fEnd) :

            if plex.getLabelValue('boundary_ids', f) == -1 : continue
            closure = plex.getTransitiveClosure(f)[0]                   # Get endpoints of edge
            crdE = []                                                   # Coordinates of the two vertices of the edge

            # Loop over closure of edge:
            for cl in closure :
                if vStart <= cl and cl < vEnd :                         # Check they are really vertices
                    off = meshd.section.getOffset(cl) / 2
                    crdE.append(mesh.coordinates.dat.data[off])

            if len(crdE) != 2 :
                print '## ERROR  number of points in crdE: %d' % len(crdE)
                exit(16)

            vn =  [crdE[0][1] - crdE[1][1], crdE[0][0] - crdE[1][0]]    # Normal vector of the edge
            nrm = sqrt(vn[0] * vn[0] + vn[1] * vn[1])                   # Norm of normal vector
            vn = [vn[0] / nrm, vn[1] / nrm]                             # Normalised normal vector

            for nid in notInDomain :

                v = nid[0]                                              # Vertex not in domain
                offnew = meshdnew.section.getOffset(v) / 2              # Corresponding section
                crdP = meshnew.coordinates.dat.data[offnew]             # Coordinates of vertex not in domain
                dst = abs(vn[0] * (crdE[0][0] - crdP[0]) +              # | vn . (crdE[0] - crdP)|
                          vn[1] * (crdE[0][1] - crdP[1]))               # = | crdE[0] - crdP | | cos(angle) |

                # Control if the vertex is between the two edge vertices (a big assumption corners are preserved):
                if dst < nid[1] :
                    # (crdP - crdE[0]) . (crdE[1] - crdE[0]) :
                    sca1 = (crdP[0] - crdE[0][0]) * (crdE[1][0] - crdE[0][0]) + \
                           (crdP[1] - crdE[0][1]) * (crdE[1][1] - crdE[0][1])
                    # (crdP - crdE[1]) . (crdE[0] - crdE[1]) :
                    sca2 = (crdP[0] - crdE[1][0]) * (crdE[0][0] - crdE[1][0]) + \
                           (crdP[1] - crdE[1][1]) * (crdE[0][1] - crdE[1][1])

                    if sca1 >= 0 and sca2 > 0 :
                        nid[1] = dst                                    # Specify (finite) distance
                        nid[2] = f                                      # Replace -1 by edge

        # Having looped over edges, loop over points not in domain (with edited notInDomain tuples):
        for nid in notInDomain :

            v = nid[0]                                          # Vertex not in domain
            offnew = meshdnew.section.getOffset(v) / 2          # Corresponding section
            crdP = meshnew.coordinates.dat.data[offnew]         # Coordinates of vertex not in domain

            if nid[1] > 0.01 :                                  # If distance is sufficiently large,

                cStart, cEnd = plex.getHeightStratum(0)         # Triangles / cells
                barCrd = [0, 0, 0]                              # Setup barycentric coordinates
                inCell = 0                                      # 0 when v is outside of cell, 1 when it is inside

                # Loop over cells:
                for c in range(cStart, cEnd) :
                    closure = plex.getTransitiveClosure(c)[0]   # Closure of triangle
                    crdC = []                                   # Coordinates of the three vertices of the triangle
                    val = []                                    # u values at vertices of the triangle

                    # Loop over entities of the triangle closure and collect u values:
                    for cl in closure :
                        if vStart <= cl and cl < vEnd :
                            off = meshd.section.getOffset(cl) / 2
                            crdC.append(mesh.coordinates.dat.data[off])
                            val.append(u.dat.data[off])

                    # Establish barycentric coordinates of v:
                    barCrd[0] = barCoord(crdP, crdC, 0)
                    barCrd[1] = barCoord(crdP, crdC, 1)
                    barCrd[2] = barCoord(crdP, crdC, 2)

                    # If v lies within the triangle,
                    if barCrd[0] >= 0 and barCrd[1] >= 0 and barCrd[2] >= 0 :
                        print 'DEBUG  Cell : %1.4f %1.4f   %1.4f %1.4f   %1.4f %1.4f   bary:  %e %e %e' % \
                              (crdC[0][0], crdC[0][1], crdC[1][0], crdC[1][1], crdC[2][0], crdC[2][1], barCrd[0],
                               barCrd[1], barCrd[2])

                        # Interpolate by expanding over barycentric basis:
                        val = barCrd[0] * val[0] + barCrd[1] * val[1] + barCrd[2] * val[2]
                        inCell = 1
                        break

                if not inCell :
                    print 'ERROR  vertex too far from the boundary but no enclosing cell found. Crd: %f %f' \
                          % (crdP[0], crdP[1])
                    exit(16)

            else :                                              # The case where a vertex is very close to the edge

                f = nid[2]                                      # Edge (which replaces -1)

                # Loop over edges:
                if f < fStart or f > fEnd :
                    print '## ERROR   f: %d,   fStart: %d,  fEnd: %d' % (f, fStart, fEnd)
                    exit(14)

                closure = plex.getTransitiveClosure(f)[0]       # Closure of edge
                crdE = []                                       # Coordinates of the two vertices of the edge
                val = []                                        # u values at the vertices of the edge

                # Loop over entities in the edge closure and collect u values:
                for cl in closure :
                    if vStart <= cl and cl < vEnd :
                        off = meshd.section.getOffset(cl) / 2
                        crdE.append(mesh.coordinates.dat.data[off])
                        val.append(u.dat.data[off])

                if len(crdE) != 2 :
                    print '## ERROR  number of points in crdE: %d' % len(crdE)
                    exit(16)

                # Normalised edge vector:
                edg =  [crdE[0][0] - crdE[1][0], crdE[0][1] - crdE[1][1]]
                nrm = sqrt(edg[0] * edg[0] + edg[1] * edg[1])
                edg = [edg[0] / nrm, edg[1] / nrm]

                # # Debugging:                                                        # TODO: Fix this!
                # print 'Coords e1 = (%1.4f, %1.4f)' % (crdE[0][0], crdE[0][1])
                # print 'Coords e2 = (%1.4f, %1.4f)' % (crdE[1][0], crdE[1][1])
                # print 'Coords p = (%1.4f, %1.4f)' % (crdP[0], crdP[1])

                # H = alpha e1 + (1-alpha) e2    and   alpha = e2P.e2e1/||e2e1||
                alpha = (crdP[0] - crdE[1][0]) * edg[0] + (crdP[1] - crdE[1][1]) * edg[1]

                # print 'alpha = ', alpha

                if alpha > 1 :
                    print '## ERROR alpha = %1.4f' % alpha
                    exit(23)
                val = alpha * val[0] + (1 - alpha) * val[1]

            unew.dat.data[offnew] = val