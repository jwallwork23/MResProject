from firedrake import *

tmesh = UnitSquareMesh(3, 3)

tV = TensorFunctionSpace(tmesh, 'CG', 1)
tM = Function(tV)
tM.interpolate(Expression([['x[0] + 1', 0], [0, 'x[1] + 1']]))

tW = FunctionSpace(tmesh, 'CG', 1)
tf = Function(tW)
tf.interpolate(Expression('x[0]'))

tfile = File('testing/testf.pvd')