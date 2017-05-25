from firedrake import *

from time import clock

from utils import adapt, construct_hessian, compute_steady_metric, interp, Meshd

# Specify problem parameters:
hmin = float(raw_input('Minimum element size (default 0.005)?: ') or 0.005)

# Define uniform mesh, with a metric function space:
mesh1 = SquareMesh(30, 30, 2, 2)
meshd1 = Meshd(mesh1)
x, y = SpatialCoordinate(mesh1)
x = x-1
y = y-1
V = TensorFunctionSpace(mesh1, 'CG', 1)
print 'Initial number of nodes : ', len(mesh1.coordinates.dat.data)

# Define sensors:
F = FunctionSpace(mesh1, 'CG', 1)
f1 = Function(F)
f2 = Function(F)
f3 = Function(F)
f4 = Function(F)
f1.interpolate(x ** 2 + y ** 2)
f2.interpolate(Expression('abs((x[0]-1) * (x[1]-1)) >= pi/25. ? \
                            0.01 * sin(50 * (x[0]-1) * (x[1]-1)) : sin(50 * (x[0]-1) * (x[1]-1))'))
f3.interpolate(0.1 * sin(50 * x) + atan(0.1 / (sin(5 * y) - 2 * x)))
f4.interpolate(atan(0.1 / (sin(5 * y) - 2 * x)) + atan(0.5 / (sin(3 * y) - 7 * x)))
f = {1: f1, 2: f2, 3: f3, 4: f4}

for i in f:

    print '*********** Sensor {y} ***********'.format(y=i)

    tic1 = clock()

    # Compute Hessian and metric:
    H = construct_hessian(mesh1, V, f[i])
    M = compute_steady_metric(mesh1, V, H, f[i], h_min = hmin)
    print 'Some element of metric field %d : ' % i
    print M.dat.data[400]

    # Adapt mesh and set up new function spaces:
    mesh2 = adapt(mesh1, M)
    meshd2 = Meshd(mesh2)
    G = FunctionSpace(mesh2, 'CG', 1)
    print 'Number of nodes after adaption : ', len(mesh2.coordinates.dat.data)

    # Interpolate functions onto new mesh:
    g = Function(G)
    interp(f[i], meshd1, g, meshd2)

    toc1 = clock()
    print 'Elapsed time adapting to sensor %d: %1.2es' % (i, toc1 - tic1)

    # Plot results:
    g.rename('Sensor {y}'.format(y=i))
    M.rename('Metric field {y}'.format(y=i))
    File('plots/adapt_plots/sensor_test{y}.pvd'.format(y=i)).write(g)
    File('plots/adapt_plots/sensor_test_metric{y}.pvd'.format(y=i)).write(M)