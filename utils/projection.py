import numpy as np
from math import radians, sin, cos

def earth_radius(lat) :
    """A function which calculates the radius of the Earth for a given latitude."""
    K = 1. / 298.257  # Earth flatness constant
    a = 6378136.3   # Semi-major axis of the Earth (m)
    return (1 - K * (sin(radians(lat)) ** 2)) * a

def lonlat2tangentxy(lon, lat, lon0, lat0):
    """A function which projects longitude-latitude coordinates onto a tangent plane at (lon0, lat0) in Cartesian coordinates
    (x,y), with units being metres."""
    Re = earth_radius(lat)
    Rphi = Re * cos(radians(lat))
    x = Rphi * sin(radians(lon - lon0))
    y = Rphi * (1 - cos(radians(lon - lon0))) * sin(radians(lat0)) + Re * sin(radians(lat - lat0))
    return x, y

def lonlat2tangent_pair(lon, lat, lon0, lat0) :
    """A function which projects longitude-latitude coordinates onto a tangent plane at (lon0, lat0) in Cartesian coordinates
    (x,y), with units being metres."""
    x, y = lonlat2tangentxy(lon, lat, lon0, lat0)
    return [x, y]

def vectorlonlat2tangentxy(lon, lat, lon0, lat0) :
    """A function which projects vectors containing longitude-latitude coordinates onto a tangent plane at (lon0, lat0) in
    Cartesian coordinates (x,y), with units being metres."""
    x = np.zeros((len(lon), 1))
    y = np.zeros((len(lat), 1))
    assert (len(x) == len(y))
    for i in range(len(x)) :
        x[i], y[i] = lonlat2tangentxy(lon[i], lat[i], lon0, lat0)
    return x, y

def mesh_converter(meshfile, lon0, lat0):
    """A function which reads in a .msh file in longitude-latitude coordinates and outputs a tangent-plane projection in
    Cartesian coordinates."""
    mesh1 = open(meshfile, 'r') # Lon-lat mesh to be converted
    mesh2 = open('resources/meshes/CartesianTohoku.msh', 'w')
    i = 0
    mode = 0
    cnt = 0
    N = -1
    for line in mesh1 :
        i += 1
        if i == 5 :
            mode += 1
        if mode == 1 :                      # Now read number
            N = int(line)                   # Number of nodes
            mode += 1
        elif mode == 2 :                    # Now edit nodes
            xy = line.split()
            xy[1], xy[2] = lonlat2tangentxy(float(xy[1]), float(xy[2]), lon0, lat0)
            xy[1] = str(xy[1])
            xy[2] = str(xy[2])
            line = ' '.join(xy)
            line += '\n'
            cnt += 1
            if cnt == N :
                assert int(xy[0]) == N      # Check all nodes have been covered
                mode += 1                   # Now the end of the nodes has been reached

        mesh2.write(line)
    mesh1.close()
    mesh2.close()

def wrapper(func, *args, **kwargs) :
    """A wrapper function to enable timing of functions with arguments"""
    def wrapped() :
        return func(*args, **kwargs)
    return wrapped
