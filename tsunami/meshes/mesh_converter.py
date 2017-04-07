#!/usr/bin/python
from math import radians, sin, cos

def earth_radius(lat):
    '''A function which calculates the radius of the Earth for a given
    latitude.'''
    K = 1./298.257  # Earth flatness constant
    a = 6378136.3  # Semi-major axis of the Earth (m)
    return (1 - K * (sin(radians(lat))**2)) * a

def lonlat2tangentxy(lon, lat, lon0, lat0):
    '''A function which projects longitude-latitude coordinates onto a
    tangent plane at (lon0, lat0) in Cartesian coordinates (x,y), with
    units being metres.'''
    Re = earth_radius(lat)
    Rphi = Re * cos(radians(lat))
    x = Rphi * sin(radians(lon-lon0))
    y = Rphi * (1 - cos(radians(lon-lon0))) * sin(radians(lat0)) + \
        Re * sin(radians(lat-lat0))
    return x, y

# Open mesh file
mesh1 = open("Tohoku_edit.msh", "r")
mesh2 = open("Cartesian_Tohoku.msh", "w")
print "Name of mesh to be edited: ", mesh1.name

i = 0
mode = 0
cnt = 0
N = -1
for line in mesh1:
    i += 1
    if (i == 5):
        mode += 1
        raw_input('Nodes located!')
    if (mode == 1):
        N = int(line)
        mode += 1
        print N
        raw_input('nodes in the mesh.')
    elif (mode == 2):
        xy = line.split()
        xy[1], xy[2] = lonlat2tangentxy(float(xy[1]), float(xy[2]), 143., 37.)
        xy[1] = str(xy[1])
        xy[2] = str(xy[2])
        line = ' '.join(xy)
        line += '\n'
        cnt += 1
    if (cnt == N):
        mode += 1
        raw_input('Nodes converted!')
        cnt +=1
    print line
    mesh2.write(line)

# Close opened files
mesh1.close()
mesh2.close()
