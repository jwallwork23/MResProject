from firedrake import *

# Set physical parameters for the schemes:
nu = 1e-3           # Viscosity (kg s^{-1} m^{-1})
g = 9.81            # Gravitational acceleration (m s^{-2})
Cb = 0.0025         # Bottom friction coefficient (dimensionless)

def nonlinear_form(u, u_, eta, eta_, v, ze, b, Dt):
    '''Weak residual form of the nonlinear shallow water equations'''
    L = (ze * (eta-eta_) - Dt * inner((eta + b) * u, grad(ze)) + inner(u-u_, v) + Dt * inner(dot(u, nabla_grad(u)), v) + \
         nu * inner(grad(u), grad(v)) + Dt * g * inner(grad(eta), v) + \
         Dt * Cb * sqrt(dot(u_, u_)) * inner(u/(eta+b), v)) * dx(degree=4)   
    return L

def linear_form(u, u_, eta, eta_, v, ze, b, Dt):
    '''Weak residual form of the linear shallow water equations'''
    L = (ze * (eta-eta_) - Dt * inner((eta + b) * u, grad(ze)) + inner(u-u_, v) + Dt * g *(inner(grad(eta), v))) * dx
    return L

def adj_nonlinear_form(w, xi):   # TODO: Needs changing!
    '''Weak residual form of the nonlinear adjoint shallow water equations'''
    L = ((le-le_) * xi - Dt * g * b * inner(lm, grad(xi)) + inner(lm-lm_, w) + Dt * inner(grad(le), w)) * dx
    return L                                                                                        # + J derivative term?

def adj_linear_form(lm, lm_, le, le_, w, xi, b, Dt):
    '''Weak residual form of the linear adjoint shallow water equations'''
    L = ((le-le_) * xi - Dt * g * b * inner(lm, grad(xi)) + inner(lm-lm_, w) + Dt * inner(grad(le), w)) * dx
    return L                                                                                        # + J derivative term?

def nonlinear_form_out(u, u_, eta, eta_, v, ze, b, Dt, mesh):
    '''Weak residual form of the nonlinear shallow water equations, with outflow boundary conditions.'''
    
    # Define the outward pointing normal to the mesh
    n = FacetNormal(mesh)
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

def linear_form_out(u, u_, eta, eta_, v, ze, b, Dt, mesh):
    '''Weak residual form of the linear shallow water equations, with outflow boundary conditions.'''
    
    # Define the outward pointing normal to the mesh
    n = FacetNormal(mesh)
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
