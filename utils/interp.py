from firedrake import *
import numpy as np

INF = float("inf")

def barCoord(crdM, crdTri, i):
   # A function which computes the barycentric coordinate of M in triangle Tri = P0, P1, P2 with respect to the ith
   # vertex crd = det(MPj, MPk) / det(PiPj, PiPk)
   j = (i+1) % 3
   k = (i+2) % 3
   res1 = (crdTri[j][0] - crdM[0]) * (crdTri[k][1] - crdM[1]) - (crdTri[k][0] - crdM[0]) * (crdTri[j][1] - crdM[1])
   res2 = (crdTri[j][0] - crdTri[i][0]) * (crdTri[k][1] - crdTri[i][1]) - \
          (crdTri[k][0] - crdTri[i][0]) * (crdTri[j][1] - crdTri[i][1])
   res = res1 / res2

   return res

def interp(u, mesh, unew, meshnew) :
    """A function which interpolates a function u onto a new mesh. Only slightly modified version of Nicolas Barral's
    function ``interpol``, from the Python script ``interpol.py``."""

    # Establish topological and geometric dimensions (usually both 2 for our purposes):
    dim = mesh._topological_dimension
    entity_dofs = np.zeros(meshnew._topological_dimension + 1, dtype = np.int32)
    entity_dofs[0] = meshnew.geometric_dimension()

    # Get the DMPlex object encapsulating the mesh topology and determine the vertices of plex to consider (?):
    plexnew = meshnew._plex
    vStart, vEnd = plexnew.getDepthStratum(0)
    
    # Establish which vertices fall outside the domain:
    notInDomain = []
    for v in range(vStart, vEnd) :
        offnew = meshnew._plex.createSection([1], entity_dofs,
                                             perm = meshnew.topology._plex_renumbering).getOffset(v) / dim
        newCrd = meshnew.coordinates.dat.data[offnew]
        try :
            val = u.at(newCrd)
        except PointNotInDomainError :
            print "####  New vertex not in domain: %f %f" % (newCrd[0], newCrd[1])
            val = 0.
            notInDomain.append([v, INF, -1])   # TODO: I should store newCrd here instead of v
        finally :
            unew.dat.data[offnew] = val
    
    # If there are points which fall outside the domain,
    if len(notInDomain) > 0 :
        print "####  Warning: number of points not in domain: %d / %d" % \
              (len(notInDomain), meshnew.topology.num_vertices())
        if mesh._topological_dimension == 2 :
            plex = mesh._plex
            fStart, fEnd = plex.getHeightStratum(1)         # Edges/facets
            vStart, vEnd = plex.getDepthStratum(0)
                    
            for f in range(fStart, fEnd) :
                if plex.getLabelValue("boundary_ids", f) == -1 : continue
                closure = plex.getTransitiveClosure(f)[0]
                crdE = []                                   # Coordinates of the two vertices of the edge
                for cl in closure:
                    if cl >= vStart and cl < vEnd : 
                        off = mesh._plex.createSection([1], entity_dofs,
                                                          perm = mesh.topology._plex_renumbering).getOffset(cl) / 2
                        crdE.append(mesh.coordinates.dat.data[off])
                if len(crdE) != 2 : exit(16)
                vn =  [crdE[0][1] - crdE[1][1], crdE[0][0] - crdE[1][0]]    # Normal vector of the edge
                nrm = sqrt(vn[0] * vn[0] + vn[1] * vn[1])
                vn = [vn[0]/nrm, vn[1]/nrm]
                for nid in notInDomain:
                    v = nid[0]
                    offnew = meshnew._plex.createSection([1], entity_dofs,
                                                         perm = meshnew.topology._plex_renumbering).getOffset(v) / 2
                    crdP = meshnew.coordinates.dat.data[offnew]
                    dst = abs(vn[0] * (crdE[0][0] - crdP[0]) + vn[1] * (crdE[0][1] - crdP[1]))

                    # Control if the vertex is between the two edge vertices (a big assumption corners are preserved):
                    if dst < nid[1]:
                        # e1P.e1e2
                        sca1 = (crdP[0] - crdE[0][0]) * (crdE[1][0] - crdE[0][0]) + \
                               (crdP[1] - crdE[0][1]) * (crdE[1][1] - crdE[0][1])
                        # e2P.e2e1
                        sca2 = (crdP[0] - crdE[1][0]) * (crdE[0][0] - crdE[1][0]) + \
                               (crdP[1] - crdE[1][1]) * (crdE[0][1] - crdE[1][1])
                        if sca1 >= 0 and sca2 > 0:
                            nid[1] = dst
                            nid[2] = f
                            
            for nid in notInDomain :
                v = nid[0]
                offnew = meshnew._plex.createSection([1], entity_dofs,
                                                     perm = meshnew.topology._plex_renumbering).getOffset(v)/2
                crdP = meshnew.coordinates.dat.data[offnew]
                val = -1
                if nid[1] > 0.01 :
                    cStart, cEnd = plex.getHeightStratum(0)
                    barCrd = [0, 0, 0]
                    inCell = 0
                    for c in range(cStart, cEnd) :
                        closure = plex.getTransitiveClosure(c)[0]
                        crdC = []           # Coordinates of the three vertices of the triangle
                        val = []            # Value of the function at the vertices of the triangle
                        for cl in closure :
                            if cl >= vStart and cl < vEnd :
                                off = mesh._plex.createSection([1], entity_dofs,
                                                            perm = mesh.topology._plex_renumbering).getOffset(cl) / 2
                                crdC.append(mesh.coordinates.dat.data[off])
                                val.append(u.dat.data[off])

                        # Establish barycentric coordinates of v:
                        barCrd[0] = barCoord(crdP, crdC, 0)
                        barCrd[1] = barCoord(crdP, crdC, 1)
                        barCrd[2] = barCoord(crdP, crdC, 2)
                        if barCrd[0] >= 0 and barCrd[1] >= 0 and barCrd[2] >= 0 :
                            print "DEBUG  Cell : %1.4f %1.4f   %1.4f %1.4f   %1.4f %1.4f   bary:  %e %e %e" % \
                                  (crdC[0][0], crdC[0][1], crdC[1][0], crdC[1][1], crdC[2][0], crdC[2][1], barCrd[0],
                                   barCrd[1], barCrd[2])
                            val = barCrd[0] * val[0] + barCrd[1] * val[1] + barCrd[2] * val[2]
                            inCell = 1
                            break
                    if not inCell :
                        print "ERROR  vertex too far from the boundary but no enclosing cell found. Crd: %f %f" \
                              % (crdP[0], crdP[1])
                        exit(16)
                else :
                    f = nid[2]
                    if (f < fStart or f > fEnd):
                        print "## ERROR   f: %d,   fStart: %d,  fEnd: %d" % (f, fStart, fEnd)
                        exit(14)
                    closure = plex.getTransitiveClosure(f)[0]
                    crdE = []                                       # Coordinates of the two vertices of the edge
                    val = []                                        # Value of the function at the vertices of the edge
                    for cl in closure:
                        if cl >= vStart and cl < vEnd : 
                            off = mesh._plex.createSection([1], entity_dofs,
                                                            perm = mesh.topology._plex_renumbering).getOffset(cl) / 2
                            crdE.append(mesh.coordinates.dat.data[off])
                            val.append(u.dat.data[off])
                    if len(crdE) != 2 : 
                        print "## ERROR  number of points in crdE: %d" % len(crdE)
                        exit(16)
                    edg =  [crdE[0][0] - crdE[1][0], crdE[0][1] - crdE[1][1]]       # Normed vector e2e1
                    nrm = sqrt(edg[0] * edg[0] + edg[1] * edg[1])
                    edg = [edg[0] / nrm, edg[1] / nrm]

                    # H = alpha e1 + (1-alpha) e2    and   alpha = e2P.e2e1/||e2e1||
                    alpha = (crdP[0] - crdE[1][0]) * edg[0] + (crdP[1] - crdE[1][1]) * edg[1]
                    if alpha > 1 : exit(23)
                    val = alpha * val[0] + (1 - alpha) * val[1]
                unew.dat.data[offnew] = val 
        else:   
            print "#### ERROR no recovery procedure implemented in 3D yet"
            exit(1)