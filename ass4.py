import numpy as np
from scipy.sparse import triu
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

# def gauss_seidel_setup(A):
#     """Splits the matrix A and returns appropriate matrices"""
#     n = A.shape[0]  # Get the size of the matrix A (n x n)
#     L = triu(A, k=1)
#     D = A - L
#     L_plus_D_inv = np.linalg.inv(D)
#     U = A - L
#     return L_plus_D_inv, U

def gauss_seidel_setup(A):
    """Splits the matrix A and returns appropriate matrices"""
    n = A.shape[0]  # Get the size of the matrix A (n x n)
    L = np.tril(A, k=-1)  # Lower triangular part
    D = np.diag(np.diag(A))  # Diagonal part
    L_plus_D_inv = np.linalg.inv(L + D)  # Inverse of (L + D)
    U = A - (L + D)  # Upper triangular part

    return L_plus_D_inv, U



def gauss_seidel_solve(L_plus_D_inv, U, b, x0=None, tol=1e-6):
    """Solves a linear system Ax=b using the Gauss-Seidel method"""
    n = b.shape[0]  
    if x0 is None:
        x_new = np.zeros(n)
    else:
        x_new = x0

    err = 2 * tol
    n_iter = 0
    TOL = 1e-6  

    while err > tol:
        x = x_new
        x_new = L_plus_D_inv @ (b - U @ x_new)
        err = norm(x_new - x) / norm(x_new)
        n_iter += 1

    return x_new, n_iter


A = np.array([[4, 1, 0, 0],
              [1, 4, 1, 0],
              [0, 1, 4, 1],
              [0, 0, 1, 4]])

b = np.array([1, 1, 1, 1])

# Setup Gauss-Seidel
L_plus_D_inv, U = gauss_seidel_setup(A)

# Solve the system 
solution_gauss_seidel, n_iter = gauss_seidel_solve(L_plus_D_inv, U, b)

# Print the solution and number of iterations
print("Gauss-Seidel Solution:", solution_gauss_seidel)
print("Number of Iterations:", n_iter)
