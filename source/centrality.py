import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def unique_upto_tol(arr, tol):
    # Sort the array
    sorted_arr = np.sort(arr)
    
    # Initialize the list for unique values
    unique_arr = []
    
    # Add the first element
    if sorted_arr.size > 0:
        unique_arr.append(sorted_arr[0])
    
    # Iterate through the sorted array
    for num in sorted_arr[1:]:
        # Compare with the last added unique number
        if num - unique_arr[-1] > tol:
            unique_arr.append(num)
    
    return np.array(unique_arr)


def nl_centrality_func(B, 
                       f=lambda x: x,
                       g=lambda x: x,
                       phi=lambda x: x,
                       psi=lambda x: x,
                       maxiter=1000, 
                       tol=1e-6, 
                       edge_weights=None,
                       node_weights=None,
                       mynorm=lambda x: np.linalg.norm(x, 1), 
                       verbose=False,
                       eigvals = True):
    """
    Computes centrality measures using SparseArrays and iterative algorithm.

    Args:
        B: Sparse adjacency matrix.
        functions: four functions
        maxiter: Maximum number of iterations.
        tol: Tolerance for convergence.
        edge_weights: Optional weights for edges (defaults to all 1).
        node_weights: Optional weights for nodes (defaults to all 1).
        mynorm: Norm function to use (defaults to L1 norm).
        verbose: If True, print progress messages during iterations.

    Returns:
        Two NumPy arrays: node centrality scores and edge centrality scores.
    """
    B = sp.csr_matrix(B)
    n, m = B.shape

    x0 = np.ones((n, 1)) / n
    y0 = np.ones((m, 1)) / m

    if edge_weights is None:
        edge_weights = np.ones((m, 1))
    if node_weights is None:
        node_weights = np.ones((n, 1))
    W = sp.diags(edge_weights.flatten())
    N = sp.diags(node_weights.flatten())

    for it in range(1, maxiter + 1):
        if verbose and (it < 10 or it % 30 == 0):
            print(f"{it} ...")
            
        u = np.sqrt(x0 * g(B @ W @ f(y0)))
        v = np.sqrt(y0 * psi(B.T @ N @ phi(x0)))
        x = u / mynorm(u)
        y = v / mynorm(v)

        check = mynorm(x - x0) + mynorm(y - y0)
        if check < tol:
            if verbose:
                print(f"{it} ===")
                x0 = x.copy()
                y0 = y.copy()
                break
            #return x.flatten(), y.flatten()

        x0 = x.copy()
        y0 = y.copy()
        rhs1 = g(B@W@f(y0))
        rhs2 = psi(B.T @ N @ phi(x0))

    node_eval = rhs1/x0
    edge_eval = rhs2/y0

    node_eval = unique_upto_tol(np.unique(node_eval[~np.isnan(node_eval)]), tol).flatten()
    edge_eval = unique_upto_tol(np.unique(edge_eval[~np.isnan(edge_eval)]), tol).flatten()
    if verbose:
        print(f"Warning: Centrality did not reach tol = {tol} in {maxiter} iterations\n------- Relative error reached so far = {check}")
    return x0.flatten(), y0.flatten(), node_eval[0], edge_eval[0]

def nonlinear_eigenvector_centrality(B, 
                                     function='linear',
                                     maxiter=1000, 
                                     tol=1e-6, 
                                     edge_weights=None,
                                     node_weights=None,
                                     mynorm=lambda x: np.linalg.norm(x, 1), 
                                     verbose=False):
    """
    Computes centrality measures using SparseArrays and iterative algorithm.

    Args:
        B: Sparse adjacency matrix.
        function: Key to select functions  (e.g., "linear", "log-exp", "max").
        maxiter: Maximum number of iterations.
        tol: Tolerance for convergence.
        edge_weights: Optional weights for edges (defaults to all 1).
        node_weights: Optional weights for nodes (defaults to all 1).
        mynorm: Norm function to use (defaults to L1 norm).
        verbose: If True, print progress messages during iterations.

    Returns:
        Two NumPy arrays: node centrality scores and edge centrality scores.
    """

    mappings = {
        "linear": (lambda x: x, lambda x: x, lambda x: x, lambda x: x),
        "log-exp": (lambda x: x, lambda x: np.power(x, 1/10), lambda x: np.log(x), lambda x: np.exp(x)),
        "max": (lambda x: x, lambda x: np.power(x, 1/5), lambda x: np.power(x, 15), lambda x: np.power(x, 1/15))
    }

    f, g, phi, psi = mappings[function]

    B = sp.csr_matrix(B)
    n, m = B.shape

    x0 = np.ones((n, 1)) / n
    y0 = np.ones((m, 1)) / m

    if edge_weights is None:
        edge_weights = np.ones((m, 1))
    if node_weights is None:
        node_weights = np.ones((n, 1))
    W = sp.diags(edge_weights.flatten())
    N = sp.diags(node_weights.flatten())

    for it in range(1, maxiter + 1):
        if verbose and (it < 10 or it % 30 == 0):
            print(f"{it} ...")
            
        u = np.sqrt(x0 * g(B @ W @ f(y0)))
        v = np.sqrt(y0 * psi(B.T @ N @ phi(x0)))
        x = u / mynorm(u)
        y = v / mynorm(v)

        check = mynorm(x - x0) + mynorm(y - y0)
        if check < tol:
            if verbose:
                print(f"{it} ===")
            return x.flatten(), y.flatten()

        x0 = x.copy()
        y0 = y.copy()

    if verbose:
        print(f"Warning: Centrality did not reach tol = {tol} in {maxiter} iterations\n------- Relative error reached so far = {check}")
    return x0.flatten(), y0.flatten()