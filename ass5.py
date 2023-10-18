import numpy as np
from scipy.io import loadmat
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve, splu  
from scipy.spatial import Delaunay
from matplotlib.tri import Triangulation
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Laplace.py code 

class LaplDiscr:
    """
    A class providing a finite element spatial discretization of the Laplace operator
    u_xx + u_yy with boundary condition K(x, y) * u + du/dn = g(t) for a given mesh.

    Example usage:
    lapl = LaplDiscr("mesh.mat")
    M, A, G = lapl.get_system()

    Attributes:
    p : NumPy array
        The x- and y-coordinates of the points in the mesh.
    e : NumPy array
        Indices of the starting and ending points of the edges of the mesh.
    t : NumPy array
        Indices of the corner points of each triangle in the mesh.
    N : int
        Number of points (degrees of freedom).

    Methods:
    get_system(g)
        Assembles and returns the discretization matrices M (mass matrix) and
        A (stiffness matrix), and the function G(t) corresponding to the ODE
        M v_t = A v + G(t) (heat equation) or
        M v_tt = A v + G(t) (wave equation) with boundary data given by g(t).
    plot_solution(v, title_str="")
        Plots the solution represented in v on the mesh. The string in title_str
        is added as the title to the plot.
    plot_mesh()
        Plots the mesh.
    """

    def __init__(self, filename):
        mesh_file = loadmat(filename)
        self.p = mesh_file['p']
        self.e = mesh_file['e']
        self.t = mesh_file['t']

        # Adjust for MATLAB indexing
        self.t[0:3, :] = self.t[0:3, :] - 1
        self.e[0:2, :] = self.e[0:2, :] - 1
        self.N = self.p.shape[1]

    def get_system(self, g):
        A = StiffMat2D(self.p, self.t)
        M = MassMat2D(self.p, self.t)
        R = RobinMat2D(self.p, self.e)
        r_inner = RobinVec2D(self.p, self.e)

        # Scale the Robin boundary data contribution by the time-dependent function g(t).
        def G(t):
            return r_inner * g(t)

        return csc_matrix(M), csc_matrix(-(A + R)), G

    def get_initial(self):
        return np.zeros(self.N)

    def plot_solution(self, v, title_str=""):
        tri = Triangulation(self.p[0, :], self.p[1, :], self.t[0:3, :].T)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_trisurf(tri, v, cmap=cm.jet, linewidth=0)
        plt.axis([-1, 1, -1, 1])
        ax.set_zlim(-1, 1)
        surf.set_clim(-1, 1)
        plt.xlabel('x')
        plt.ylabel('y')
        fig.colorbar(mappable=surf)
        plt.title(title_str)
        plt.show()

    def plot_mesh(self):
        tri = Triangulation(self.p[0, :], self.p[1, :], self.t[0:3, :].T)
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.triplot(tri)
        plt.axis([-1, 1, -1, 1])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

# Functions for assembling the matrices. Taken from
# Larson, M.G., Bengzon, F. (2013). The Finite Element.
# Gauss-Seidel setup function
def gauss_seidel_setup(A):
    n = A.shape[0]
    
    # Convert A to a dense array
    A_dense = A.toarray()
    
    L = A_dense.copy()
    L[np.triu_indices(n, k=1)] = 0  # Set upper triangular elements to zero
    
    # Use diag on the dense array
    D = np.diag(np.diag(A_dense))
    
    L_plus_D_inv = np.linalg.inv(L + D)
    U = A_dense - (L + D)
    
    return L_plus_D_inv, U

# Gauss-Seidel solve function
def gauss_seidel_solve(L_plus_D_inv, U, b, x0=None, tol=1e-6):
    n = b.shape[0]
    if x0 is None:
        x_new = np.zeros(n)
    else:
        x_new = x0

    err = 2 * tol

    while err > tol:
        x = x_new
        x_new = L_plus_D_inv @ (b - U @ x_new)
        err = np.linalg.norm(x_new - x) / np.linalg.norm(x_new)

    return x_new
# Conjugate Gradient solve function
def conjugate_gradient_solve(A, b, x0=None, tol=1e-6):
    if x0 is None:
        x_new = np.zeros(b.shape)
    else:
        x_new = x0

    err = 2 * tol
    n_iter = 0
    TOL = 1e-6

    r = b - A @ x_new
    p = r

    while err > tol:
        x = x_new
        Ap = A @ p
        alpha = np.dot(r, r) / np.dot(p, Ap)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        err = np.linalg.norm(x_new - x) / np.linalg.norm(x_new)
        r = r_new
        n_iter += 1

    return x_new, n_iter
def RobinMat2D(p, e):
    Np = p.shape[1]
    Ne = e.shape[1]
    R = np.zeros((Np, Np))

    for E_idx in range(Ne):
        loc2glb = e[0:2, E_idx].astype(int)
        x = p[0, loc2glb]
        y = p[1, loc2glb]
        length = np.sqrt((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2)
        RK = K(x, y) / 6 * np.array([[2, 1], [1, 2]]) * length
        mat_idx = np.ix_(loc2glb.tolist(), loc2glb.tolist())
        R[mat_idx] = R[mat_idx] + RK

    return R

def RobinVec2D(p, e):
    Np = p.shape[1]
    Ne = e.shape[1]
    r = np.zeros(Np)

    for E_idx in range(Ne):
        loc2glb = e[0:2, E_idx].astype(int)
        x = p[0, loc2glb]
        y = p[1, loc2glb]
        length = np.sqrt((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2)
        rK = K(x, y) * np.array([1, 1]) * length / 2
        r[loc2glb] = r[loc2glb] + rK

    return r

def StiffMat2D(p, t):
    Np = p.shape[1]
    Nt = t.shape[1]
    A = np.zeros((Np, Np))

    for T_idx in range(Nt):
        loc2glb = t[0:3, T_idx].astype(int)
        x = p[0, loc2glb]
        y = p[1, loc2glb]
        area = np.abs(np.sum((x[[1, 2, 0]] - x) * (y[[1, 2, 0]] + y))) / 2
        b = np.array([y[1] - y[2], y[2] - y[0], y[0] - y[1]]) / (2 * area)
        c = np.array([x[2] - x[1], x[0] - x[2], x[1] - x[0]]) / (2 * area)
        AK = (np.tensordot(b, b, axes=0) + np.tensordot(c, c, axes=0)) * area
        mat_idx = np.ix_(loc2glb.tolist(), loc2glb.tolist())
        A[mat_idx] = A[mat_idx] + AK

    return A

def MassMat2D(p, t):
    Np = p.shape[1]
    Nt = t.shape[1]
    M = np.zeros((Np, Np))

    for T_idx in range(Nt):
        loc2glb = t[0:3, T_idx].astype(int)
        x = p[0, loc2glb]
        y = p[1, loc2glb]
        area = np.abs(np.sum((x[[1, 2, 0]] - x) * (y[[1, 2, 0]] + y))) / 2
        MK = area * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) / 12
        mat_idx = np.ix_(loc2glb.tolist(), loc2glb.tolist())
        M[mat_idx] = M[mat_idx] + MK

    return M

# Define the boundary condition function K(x, y)
def K(x, y):
    outer_circ_rad = 1
    if np.any(np.abs(np.sqrt(x**2 + y**2) - outer_circ_rad) < 1e-12):
        return 0
    else:
        return 1

# Function to solve the heat equation using various methods
def solve_heat_equation(mesh, T, delta_t, method="direct"):
    # Load the mesh
    mesh_obj = LaplDiscr(mesh)

    # Define the boundary function g(t)
    def g(t):
        return np.sin(4 * np.pi * t)

    # Assemble the discretization matrices and boundary function
    M, A, G = mesh_obj.get_system(g)

    # Initialize v with zeros at t = 0
    v = mesh_obj.get_initial()

    # Time-stepping loop
    for t in np.arange(0, T, delta_t):
        if method == "gaussian_elemination":
            v = spsolve(M - delta_t * A, M.dot(v) + delta_t * G(t))
        elif method == "gauss_seidel":
            L_plus_D_inv, U = gauss_seidel_setup(M - delta_t * A)
            v = gauss_seidel_solve(L_plus_D_inv, U, M.dot(v) + delta_t * G(t))
        elif method == "conjugate_gradient":
            v, _ = conjugate_gradient_solve(M - delta_t * A, M.dot(v) + delta_t * G(t), x0=v)
        elif method == "lu_decomposition":
            lu = splu(M - delta_t * A)
            v = lu.solve(M.dot(v) + delta_t * G(t))
    
    return v

def main():
    # Mesh resolutions
    mesh_resolutions = ["mesh.mat", "liz.mat"]  # List of mesh files

    # Set parameters
    T = 1.0
    delta_t = 0.001

    methods = ["gaussian_elemination", "gauss_seidel", "conjugate_gradient", "lu_decomposition"]

    for mesh_file in mesh_resolutions:
        print(f"Mesh Resolution: {mesh_file}")
        for method in methods:
            print(f"Method: {method}")
        
            start_time = time.time()
            v = solve_heat_equation(mesh_file, T, delta_t, method)
            end_time = time.time()
        
            print(f"Solution: {v}")
            print(f"Time taken: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    main()