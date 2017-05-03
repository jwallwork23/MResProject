from firedrake import *
import numpy as np
from numpy import linalg as LA

from adaptivity import *
from interp import *

# Define original mesh, with a metric function space:
mesh1 = SquareMesh(30, 30, 2, 2)
x, y = SpatialCoordinate(mesh1)
V1 = TensorFunctionSpace(mesh1, 'CG', 1)
M = Function(V1)

# Define sensors:
F = FunctionSpace(mesh1, 'CG', 1)
f1 = Function(F); f2 = Function(F); f3 = Function(F); f4 = Function(F)
f1.interpolate((x-1)**2 + (y-1)**2)
f2.interpolate(Expression('abs((x[0]-1)*(x[1]-1)) >= pi/25. ? 0.01*sin(50*(x[0]-1)*(x[1]-1)) : sin(50*(x[0]-1)*(x[1]-1))'))
f3.interpolate(0.1*sin(50*(x-1)) + atan(0.1/(sin(5*(y-1))-2*(x-1))))
f4.interpolate(atan(0.1/(sin(5*(y-1))-2*(x-1))) + atan(0.5/(sin(3*(y-1))-7*(x-1))))

f = {1: f1, 2: f2, 3: f3, 4: f4}
eps = {1: 0.02, 2: 1., 3: 1., 4: 0.2}

for i in f:

    # Compute Hessian and metric:
    H, V = construct_hessian(mesh1, f[i])
    M = compute_steady_metric(mesh1, V, H, f[i], eps[i])

    # Adapt mesh and set up new function spaces:
    mesh2 = adapt(mesh1, M)
    G = FunctionSpace(mesh2, 'CG', 1)

    # Interpolate functions onto new mesh:
    g = Function(G)
    interp(f[i], mesh1, g, mesh2)

    # Plot results:
    g.rename('Sensor {y}'.format(y=i))
    File('plots/adapt_plots/sensor_test{y}a.pvd'.format(y=i)).write(f[i])
    File('plots/adapt_plots/sensor_test{y}b.pvd'.format(y=i)).write(g)
