from firedrake import *

# Set physical parameters for the schemes:
nu = 1e-3           # Viscosity (kg s^{-1} m^{-1})
g = 9.81            # Gravitational acceleration (m s^{-2})
Cb = 0.0025         # Bottom friction coefficient (dimensionless)

def linear_form_1d(mu, mu_, eta, eta_, nu, ze, b, Dt):
    """Weak residual form of the 1D linear shallow water equations in momentum form."""
    L = ((eta-eta_) * ze - Dt * mu * ze.dx(0) + (mu-mu_) * nu + Dt * g * b * eta.dx(0) * nu) * dx
    return L

def linear_form_2d(mu, mu_, eta, eta_, nu, ze, b, Dt):
    """Weak residual form of the 2D linear shallow water equations in momentum form."""
    L = ((eta-eta_) * ze - Dt * inner(mu, grad(ze)) + inner(mu-mu_, nu) + Dt * g * b * (inner(grad(eta), nu))) * dx
    return L

def nonlinear_form(u, u_, eta, eta_, v, ze, b, Dt):
    """Weak residual form of the nonlinear shallow water equations."""
    L = (ze * (eta-eta_) - Dt * inner((eta + b) * u, grad(ze)) + inner(u-u_, v) + Dt * inner(dot(u, nabla_grad(u)), v) +
        nu * inner(grad(u), grad(v)) + Dt * g * inner(grad(eta), v) +
        Dt * Cb * sqrt(dot(u_, u_)) * inner(u/(eta+b), v)) * dx(degree=4)
    return L

def linear_form(u, u_, eta, eta_, v, ze, b, Dt):
    """Weak residual form of the linear shallow water equations."""
    L = (ze * (eta-eta_) - Dt * inner((eta + b) * u, grad(ze)) + inner(u-u_, v) + Dt * g *(inner(grad(eta), v))) * dx
    return L

def adj_linear_form_1d(lm, lm_, le, le_, v, w, b, Dt):
    """Weak residual form of the 1D linear adjoint shallow water equations in momentum form."""
    L = ((le-le_) * w + Dt * g * b * lm * w.dx(0) + (lm-lm_) * v - Dt * le.dx(0) * v) * dx
    return L

def adj_linear_form_2d(lm, lm_, le, le_, w, xi, b, Dt):
    """Weak residual form of the 2D linear adjoint shallow water equations in momentum form."""
    L = ((le-le_) * xi - Dt * g * b * inner(lm, grad(xi)) + inner(lm-lm_, w) + Dt * inner(grad(le), w)) * dx
    return L
                                                                                       # + J derivative term?
def adj_nonlinear_form(w, xi):   # TODO: Needs changing!
    """Weak residual form of the nonlinear adjoint shallow water equations."""
    L = ((le-le_) * xi - Dt * g * b * inner(lm, grad(xi)) + inner(lm-lm_, w) + Dt * inner(grad(le), w)) * dx
    return L                                                                                        # + J derivative term?

def adj_linear_form(lm, lm_, le, le_, w, xi, b, Dt):
    """Weak residual form of the linear adjoint shallow water equations."""
    L = ((le-le_) * xi - Dt * g * b * inner(lm, grad(xi)) + inner(lm-lm_, w) + Dt * inner(grad(le), w)) * dx
    return L                                                                                        # + J derivative term?

def nonlinear_form_out(u, u_, eta, eta_, v, ze, b, Dt, n):
    """Weak residual form of the nonlinear shallow water equations, with outflow boundary conditions."""

    # Integrate terms of the momentum equation over the interior:
    Lu_int = (inner(u-u_, v) + Dt * (inner(dot(u, nabla_grad(u)), v) + nu * inner(grad(u), grad(v)) + g * inner(grad(eta), v))
    + Dt * Cb * sqrt(dot(u_, u_)) * inner(u / (eta + b), v)) * dx(degree=4)
    # Integrate terms of the continuity equation over the interior:
    Le_int = (ze * (eta-eta_) - Dt * inner((eta + b) * u, grad(ze))) * dx(degree=4)
    # Integrate over left-hand boundary:
    L_side1 = Dt * (-inner(dot(n, nabla_grad(u)), v) + dot(u, n) * (ze * (eta + b))) * ds(1)(degree=4)
    # Integrate over right-hand boundary:
    L_side2 = Dt * (-inner(dot(n, nabla_grad(u)), v) + dot(u, n) * (ze * (eta + b))) * ds(2)(degree=4)
    # Establish the bilinear form using the above integrals:
    return Lu_int + Le_int + L_side1 + L_side2

def linear_form_out(u, u_, eta, eta_, v, ze, b, Dt, n):
    """Weak residual form of the linear shallow water equations, with outflow boundary conditions."""

    # Integrate terms of the momentum equation over the interior:
    Lu_int = (inner(u-u_, v) + Dt * g *(inner(grad(eta), v))) * dx
    # Integrate terms of the continuity equation over the interior:
    Le_int = (ze * (eta-eta_) - Dt * inner((eta + b) * u, grad(ze))) * dx
    # Integrate over left-hand boundary:
    L_side1 = dot(u, n) * (ze * (eta + b)) * ds(1)
    # Integrate over right-hand boundary:
    L_side2 = dot(u, n) * (ze * (eta + b)) * ds(2)
    # Establish the bilinear form using the above integrals:

    return Lu_int + Le_int + L_side1 + L_side2

def SW_solve(q_, q, u_, eta_, b, Dt, Vq, params, form, BCs=[], n=None):
    """A function which solves shallow water type problems."""

    # Build the weak form of the timestepping algorithm, expressed as a mixed nonlinear problem:
    v, ze = TestFunctions(Vq)
    u, eta = split(q)
    u_, eta_ = split(q_)

    # Establish form:
    L = form(u, u_, eta, eta_, v, ze, b, Dt, n)

    # Set up the variational problem
    q_prob = NonlinearVariationalProblem(L, q, bcs=BCs)
    q_solv = NonlinearVariationalSolver(q_prob, solver_parameters = params)

    # The function 'split' has two forms: now use the form which splits a function in order to access its data
    u_, eta_ = q_.split(); u, eta = q.split()

    # Store multiple functions
    u.rename('Fluid velocity')
    eta.rename('Free surface displacement')

    return q_, q, u_, u, eta_, eta, q_solv
