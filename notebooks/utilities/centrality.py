import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


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
                       verbose=False):
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
            return x.flatten(), y.flatten()

        x0 = x.copy()
        y0 = y.copy()

    if verbose:
        print(f"Warning: Centrality did not reach tol = {tol} in {maxiter} iterations\n------- Relative error reached so far = {check}")
    return x0.flatten(), y0.flatten()


def hevc(B, 
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


def logexp_hevc(B, 
               maxiter=1000, 
               tol=1e-6, 
               edge_weights=None,
               node_weights=None,
               mynorm=lambda x: np.linalg.norm(x, 1), 
               verbose=False):
    """
    Computes log-exp eigenvector centrality for bipartite graphs.

    Args:
        B: Sparse adjacency matrix.
        maxiter: Maximum number of iterations.
        tol: Tolerance for convergence.
        edge_weights: Optional weights for edges (defaults to all 1).
        node_weights: Optional weights for nodes (defaults to all 1).
        mynorm: Norm function to use (defaults to L1 norm).
        verbose: If True, print progress messages during iterations.

    Returns:
        Two NumPy arrays: node centrality scores and edge centrality scores.
    """

    # log-exp transformation functions
    f   = lambda x: x
    g   = lambda x: np.power(x, 1/10)
    phi = lambda x: np.log(x)
    psi = lambda x: np.exp(x)

    B = sp.csr_matrix(B)
    n, m = B.shape

    # Initial values
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

        x0, y0 = x.copy(), y.copy()

    if verbose:
        print(f"Warning: Did not converge to tol={tol} in {maxiter} iterations "
              f"(final error={check})")
    return x0.flatten(), y0.flatten()


def linear_hevc(B,
               maxiter=1000,
               tol=1e-6,
               edge_weights=None,
               node_weights=None,
               mynorm=lambda x: np.linalg.norm(x, 1),
               verbose=False):
    """
    Linear eigenvector-like centrality on a bipartite incidence matrix B (n x m).
    Returns (node_scores, edge_scores).
    """
    B = sp.csr_matrix(B)
    n, m = B.shape

    x0 = np.ones((n, 1)) / n
    y0 = np.ones((m, 1)) / m

    if edge_weights is None:
        edge_weights = np.ones((m, 1))
    if node_weights is None:
        node_weights = np.ones((n, 1))

    W = sp.diags(edge_weights.ravel())
    N = sp.diags(node_weights.ravel())

    eps = 1e-15
    for it in range(1, maxiter + 1):
        if verbose and (it < 10 or it % 30 == 0):
            print(f"{it} ...")

        # Linear case: f=g=phi=psi=identity
        u_raw = x0 * (B @ W @ y0)
        v_raw = y0 * (B.T @ N @ x0)

        # Guard sqrt / norms
        u = np.sqrt(np.maximum(u_raw, 0.0))
        v = np.sqrt(np.maximum(v_raw, 0.0))

        nu = mynorm(u) + eps
        nv = mynorm(v) + eps
        x = u / nu
        y = v / nv

        err = mynorm(x - x0) + mynorm(y - y0)
        if err < tol:
            if verbose:
                print(f"{it} ===")
            return x.ravel(), y.ravel()

        x0, y0 = x, y

    if verbose:
        print(f"Warning: did not converge to tol={tol} in {maxiter} iters (err={err})")
    return x0.ravel(), y0.ravel()