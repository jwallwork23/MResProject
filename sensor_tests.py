from firedrake import *
import numpy as np
from time import clock

import utils.adaptivity as adap
import utils.interp as inte

print('\n******************************** SENSOR ADAPTIVITY TESTS ********************************\nOptions...')
print('f1(x,y) = x^2 + y^2\n')
print('f2(x,y) = / 0.01 * sin(50(x-1)(y-1))     if      abs((x-1)(y-1)) >= pi/25,')
print('          \ sin(50(x-1)(y-1))            else\n')
print('f3(x,y) = 0.1 * sin(50x) + atan(0.1 / (sin(5y) - 2x))\n')
print('f4(x,y) = atan(0.1 / (sin(5y) - 2x)) + atan(0.5 / (sin(3y) - 7x))\n')

# Specify problem parameters:
choices = int(input('Choose sensor (1/2/3/4 or 0 to try all): ') or 0)
hmin = float(input('Minimum element size (default 0.0001)?: ') or 0.0001)
ani = float(input('Maximum aspect ratio (default 100.)?: ') or 100.)
num = int(input('Number of adaptations (default 4)?: ') or 4)
ntype = input('Normalisation type? (lp/manual): ') or 'lp'
if ntype not in ('lp', 'manual'):
    raise ValueError('Please try again, choosing lp or manual.')
iso = bool(input('Hit anything but enter to use isotropic, rather than anisotropic: ')) or False
if not iso:
    hess_meth = input('Integration by parts or double L2 projection? (parts/dL2, default dL2): ') or 'dL2'
    if hess_meth not in ('parts', 'dL2'):
        raise ValueError('Please try again, choosing parts or dL2.')
print('\n')

for i in range(1, 5):
    if choices in (i, 0):
        print('******************************** Sensor %d ********************************' % i, '\n')

        # Define uniform mesh, with a metric function space:
        mesh = SquareMesh(200, 200, 2, 2)
        x, y = SpatialCoordinate(mesh)
        x = x - 1
        y = y - 1
        print('Initial number of nodes : ', len(mesh.coordinates.dat.data))

        # Interpolate sensor field:
        W = FunctionSpace(mesh, 'CG', 1)
        f = Function(W, name='Sensor {y}'.format(y=i))
        if i == 1:
            f.interpolate(x ** 2 + y ** 2)
        elif i == 2:
            f.interpolate(Expression('abs((x[0]-1) * (x[1]-1)) >= pi/25. ? \
                                      0.01 * sin(50 * (x[0]-1) * (x[1]-1)) : sin(50 * (x[0]-1) * (x[1]-1))'))
        elif i == 3:
            f.interpolate(0.1 * sin(50 * x) + atan(0.1 / (sin(5 * y) - 2 * x)))
        elif i == 4:
            f.interpolate(atan(0.1 / (sin(5 * y) - 2 * x)) + atan(0.5 / (sin(3 * y) - 7 * x)))

        # Set up output files:
        if iso:
            f_file = File('plots/isotropic_outputs/sensor_test{y}.pvd'.format(y=i))
            M_file = File('plots/isotropic_outputs/sensor_test_metric{y}.pvd'.format(y=i))
            H_file = File('plots/isotropic_outputs/sensor_test_hessian{y}.pvd'.format(y=i))
        else:
            f_file = File('plots/anisotropic_outputs/sensor_test{y}.pvd'.format(y=i))
            M_file = File('plots/anisotropic_outputssensor_test_metric{y}.pvd'.format(y=i))
            H_file = File('plots/anisotropic_outputs/sensor_test_hessian{y}.pvd'.format(y=i))

        for j in range(num):
            tic1 = clock()

            # Compute Hessian and metric:
            V = TensorFunctionSpace(mesh, 'CG', 1)
            H = Function(V)
            if iso:
                for i in range(len(H.dat.data)):
                    H.dat.data[i][0, 0] = np.abs(f.dat.data[i])
                    H.dat.data[i][1, 1] = np.abs(f.dat.data[i])
            else:
                H = adap.construct_hessian(mesh, V, f, method=hess_meth)
            M = adap.compute_steady_metric(mesh, V, H, f, h_min=hmin, a=ani, normalise=ntype)

            # Adapt mesh and interpolate functions:
            adaptor = AnisotropicAdaptation(mesh, M)
            mesh = adaptor.adapted_mesh
            fields = inte.interp(mesh, f)

            W = FunctionSpace(mesh, 'CG', 1)
            f = Function(W, name='Sensor {y}'.format(y=i))
            f.dat.data[:] = fields[0].dat.data[:]
            toc1 = clock()

            # Relabel functions:
            M.rename('Metric field {y}'.format(y=i))
            H.rename('Hessian {y}'.format(y=i))

            # Print to screen:
            print('\nNumber of nodes after adaption %d : %d ' % (j + 1, len(mesh.coordinates.dat.data)))
            print('Elapsed time for this step: %1.2es' % (toc1 - tic1), '\n')

            # Plot results:
            f_file.write(f, time=j)
            M_file.write(M, time=j)
            H_file.write(H, time=j)
