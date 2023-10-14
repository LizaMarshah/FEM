import numpy as np
from numpy.linalg import norm

def conjugate_gradient_solve(A, b, x0=None, tol=1e-6):
    """Solves a linear system Ax=b using the conjugate gradient method"""
    # A: left-hand side matrix
    # b: right-hand side vector
    # x0: Initial guess (if None, the zero vector is used)
    # tol: Relative error tolerance
    
    # If no initial guess supplied, use the zero vector
    if x0 is None:
        x_new = np.zeros_like(b)
    else:
        x_new = x0

    # r: residual
    # p: search direction
    r = b - A @ x_new
    rho = norm(r) ** 2
    p = np.copy(r)
    err = 2 * tol
    n_iter = 0

    while err > tol:
        x = x_new
        w = A @ p
        Anorm_p_squared = np.dot(p, w)
        
        # If norm_A(p) is 0, we should have converged.
        if Anorm_p_squared == 0:
            break

        alpha = rho / Anorm_p_squared
        x_new = x + alpha * p
        r -= alpha * w
        rho_prev = rho
        rho = norm(r) ** 2
        p = r + (rho / rho_prev) * p
        err = norm(x_new - x) / norm(x_new)
        n_iter += 1

    return x_new, n_iter

def gauss_seidel_setup(A):
    """Splits the matrix A and returns appropriate matrices"""
    # More code here ...
    # Return matrix decomposition
    return L_plus_D_inv, U

def gauss_seidel_solve(L_plus_D_inv, U, b, x0=None, tol=1e-6):
    """Solves a linear system Ax=b using the Gauss-Seidel method"""
    # L_plus_D_inv: inv(L+D) (inverse of lower triangular parts and diagonal of A)
    # U: upper triangular part of A
    # b: right-hand side vector
    # x0: Initial guess (if None, the zero vector is used)
    # tol: Relative error tolerance
    
    # If no initial guess supplied, use the zero vector
    if x0 is None:
        x_new = np.zeros_like(b)
    else:
        x_new = x0

    # More code here ...
    # Return approximate solution
    return x
