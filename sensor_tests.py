from firedrake import *

from time import clock

from utils import adapt, construct_hessian, compute_steady_metric, interp, Meshd

# Specify problem parameters:
choices = int(raw_input('Choose sensor (1/2/3/4 or hit enter to try all): ')) or 0
hmin = float(raw_input('Minimum element size (default 0.0001)?: ') or 0.0001)
ani = float(raw_input('Maximum aspect ratio (default 100.)?: ') or 100.)
N = int(raw_input('Number of adaptions (default 4)?: ') or 4)

# Define uniform mesh, with a metric function space:
mesh1 = SquareMesh(200, 200, 2, 2)
meshd1 = Meshd(mesh1)
x, y = SpatialCoordinate(mesh1)
x = x - 1
y = y - 1
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
print ''

for i in f :

    if choices in (i, 0) :

        mesh = mesh1
        V = TensorFunctionSpace(mesh1, 'CG', 1)

        print '*********** Sensor %d ***********' % i

        for j in range(N) :
        
            tic1 = clock()

            # Compute Hessian and metric:
            H = construct_hessian(mesh, V, f[i])
            M = compute_steady_metric(mesh, V, H, f[i], h_min = hmin, a = ani)

            # Adapt mesh and set up new function spaces:
            mesh_ = mesh
            mesh = adapt(mesh, M)
            meshd_ = Meshd(mesh_)
            meshd = Meshd(mesh)
            G = FunctionSpace(mesh, 'CG', 1)

            # Interpolate functions onto new mesh:
            g = Function(G)
            interp(f[i], meshd_, g, meshd)
            V = TensorFunctionSpace(mesh, 'CG', 1)
            f[i] = g
            toc1 = clock()

            # Print to screen:
            print ''
            print 'Number of nodes after adaption %d : %d ' % (j, len(mesh.coordinates.dat.data))
            print 'Elapsed time for this step: %1.2es' % (toc1 - tic1)
            print ''

            # Plot results:
            g.rename('Sensor {y}'.format(y = i))
            M.rename('Metric field {y}'.format(y = i))
            File('plots/adapt_plots/sensor_test{y}.pvd'.format(y = i)).write(g)
            File('plots/adapt_plots/sensor_test_metric{y}.pvd'.format(y = i)).write(M)