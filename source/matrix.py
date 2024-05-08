import pandas as pd
import numpy as np
from scipy.linalg import toeplitz
import scipy.sparse as sps
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from scipy.stats import chi2


def larntzPerlman(M1, M2, sample_size, alpha=0.05):
    """Larntz-Perlman procedure for testing correlation matrix equivalence (two matrices).
    
    NOTES:
         - Null Hypothesis (H0):  This is the starting assumption that the two correlation matrices
         represent the same underlying phenomenon. In simpler terms, they depict similar patterns of
         correlations between variables.
         
         - Rejecting H0:  The Larntz-Perlman procedure uses statistical tests to assess how likely it 
         is that the observed differences between the matrices could have occurred by random chance. 
         If the test statistic (maximum S value) is extreme compared to what's expected under the null
         hypothesis (unlikely due to random variation), we reject H0.

    Args:
        M1 (numpy.ndarray): The first correlation matrix (p x p).
        M2 (numpy.ndarray): The second correlation matrix (p x p).
        sample_size (int): Sample size.
        alpha (float): Alpha parameter for significance determination (default: 0.05).

    Returns:
        tuple:
            - hypothesis_accepted (bool): True if the null hypothesis is accepted.
                False: reject, data cannot be pooled
                True: accept, data can be pooled
            - p_values (numpy.ndarray): Matrix of p-values (uncorrected).
            - s_matrix (numpy.ndarray): S matrix.
            - overall_p_value (float): Overall p-value.
    """

    num_variables = M1.shape[0]  # Assumes both matrices have the same dimensions

    # Fisher Z transform (extract upper triangle elements)
    fisher_z_values = np.arctanh(np.array([
        M1[np.triu_indices(num_variables, k=1)],
        M2[np.triu_indices(num_variables, k=1)]
    ]))

    # Calculate mean Z score, S, and test statistic
    mean_z_score = np.mean(fisher_z_values, axis=0)  # Mean across matrices
    s_values = (sample_size - 3) * np.sum((fisher_z_values - mean_z_score) ** 2, axis=0) 
    test_statistic = np.max(s_values)

    # Sidak correction and null hypothesis testing (degrees of freedom = 1)
    sidak_alpha = (1 - alpha) ** (2 / (num_variables * (num_variables - 1)))
    hypothesis_accepted = test_statistic <= chi2.ppf(sidak_alpha, 1) 

    # Format output (same logic as before)
    p_values = np.zeros((num_variables, num_variables))
    p_values[np.triu_indices(num_variables, k=1)] = 1 - chi2.cdf(s_values, 1) 
    p_values += p_values.T

    s_matrix = np.zeros((num_variables, num_variables))
    s_matrix[np.triu_indices(num_variables, k=1)] = s_values
    s_matrix += s_matrix.T

    overall_p_value = 1 - (chi2.cdf(test_statistic, 1)**((num_variables * (num_variables - 1)) / 2))

    return hypothesis_accepted, p_values, s_matrix, overall_p_value


def handle_outliers(A, n):
    """Identifies and modifies outliers in a matrix.

    Args:
        A (numpy.ndarray): The input matrix.
        n (int): The number of top outliers to handle.

    Returns:
        numpy.ndarray: The modified matrix with outliers replaced.
    """

    # Ensure you don't exceed the number of elements in the matrix
    n = min(n, A.size)  

    row_idx, col_idx = get_sorted_upper_triangle_indices(A)

    # Update outliers 
    for i in range(n):
        A[row_idx[i], col_idx[i]] = A.mean()  # Replace outliers with the matrix mean

    return A


def get_sorted_upper_triangle_indices(matrix, descending=True):
    """Returns sorted indices of the upper triangle of a matrix, based on their values.

    Args:
        matrix (np.ndarray): The input matrix.
        descending (bool, optional): If True, sorts in descending order. 
                                     Defaults to True.

    Returns:
        list: A list of tuples where each tuple represents a sorted index pair 
              (row, column) in the upper triangle.
    """
    sorted_idx = np.unravel_index(np.argsort(-1*matrix, axis=None), matrix.shape)
    row_idx, col_idx = sorted_idx
    row_idx = np.array(row_idx).ravel()
    col_idx = np.array(col_idx).ravel()
    return row_idx, col_idx


def remove_indices(arr, indices):
    """Removes rows and columns from a square array based on a list of indices.

    Args:
      arr: A square numpy array.
      indices: A list of indices to remove.

    Returns:
      A new numpy array with the specified rows and columns removed.
    """
    valid_indices = np.asarray([i for i in range(len(arr)) if i not in indices])
    return arr[np.ix_(valid_indices, valid_indices)]


def normalize_oe(matrix):
    """Normalizes a symmetric matrix by its Toeplitz expectation. 
    Optimizes calculations assuming symmetry. 

    Args:
        matrix (np.ndarray): The input symmetric matrix to be normalized.

    Returns:
        np.ndarray: The normalized matrix.
    """
    def calculate_diagonal_means(matrix):
        """Calculates the mean values from the upper triangular diagonals."""
        diag_means = []
        for offset in range(matrix.shape[0]):  # Only iterate up to the main diagonal
            diag_means.append(np.mean(np.diagonal(matrix, offset=offset)))
        return diag_means

    diagonal_means = calculate_diagonal_means(matrix)
    toeplitz_matrix = toeplitz(diagonal_means)  # Toeplitz matrices are symmetric

    normalized_matrix = np.divide(matrix, toeplitz_matrix)
    np.nan_to_num(normalized_matrix, copy=False, nan=0.0)
    return normalized_matrix


def convert_to_csr(data):
    """
    Converts NumPy arrays, NumPy matrices, or Pandas DataFrames to SciPy CSR matrices.

    Args:
        data (np.ndarray, np.matrix, or pd.DataFrame): The input data to be converted.

    Returns:
        scipy.sparse.csr_matrix: The converted CSR matrix.

    Raises:
        TypeError: If the input data type is not supported.
    """

    if isinstance(data, np.ndarray):
        # For NumPy arrays:
        return csr_matrix(data)

    elif isinstance(data, np.matrix):
        # For NumPy matrices:
        return csr_matrix(data.A)  # Convert to a regular array first

    elif isinstance(data, pd.DataFrame):
        # For Pandas DataFrames:
        return csr_matrix(data.values)  # Convert to a NumPy array

    else:
        raise TypeError("Unsupported data type. Please provide a NumPy array, NumPy matrix, or Pandas DataFrame.")


def normalize_kr(A, tol=1e-6, max_outer_iterations=30, max_inner_iterations=10):
    """
    adapted from: https://github.com/ay-lab/HiCKRy/blob/master/Scripts/knightRuiz.py
    
    KnightRuizAlg is an implementation of the matrix balancing algorithm developed by Knight and Ruiz.
    The goal is to take a matrix A and find a vector x such that diag(x)*A*diag(x) returns a doubly stochastic matrix.

    :param A: input array
    :param tol: error tolerance
    :param max_outer_iterations: maximum number of outer iterations
    :param max_inner_iterations: maximum number of inner iterations by CG
    :return: Ahat the normalized matrix
    """
    A = convert_to_csr(A)

    n = A.shape[0]  # Get the size of the input matrix
    e = np.ones((n, 1), dtype=np.float64)  # Create a vector of ones
    res = []

    Delta = 3  # Cone boundary value
    delta = 0.1  # Cone boundary value
    x0 = np.copy(e)  # Initial guess for the balancing vector
    g = 0.9  # Damping factor

    etamax = eta = 0.1  # Initialization of the inner iteration step size
    stop_tol = tol * 0.5  # Tolerance for stopping inner iterations
    x = np.copy(x0)  # Copy the initial guess

    rt = tol ** 2.0  # Square of the error tolerance
    v = x * (A.dot(x))  # Pre-calculate xAx
    rk = 1.0 - v  # Calculate residual vector
    rho_km1 = ((rk.transpose()).dot(rk))[0, 0]  # Calculate squared norm of the residual vector
    rho_km2 = rho_km1  # Store the previous squared norm of the residual vector
    rout = rold = rho_km1  # Initialize outer iteration residuals

    MVP = 0  # Counter for matrix-vector products
    i = 0  # Outer iteration count

    # Outer iteration loop
    while rout > rt and i < max_outer_iterations:
        i += 1
        k = 0
        y = np.copy(e)  # Initialize search direction
        innertol = max(eta ** 2.0 * rout, rt)  # Calculate inner iteration tolerance

        # Inner iteration loop by CG method
        while rho_km1 > innertol and k < max_inner_iterations:
            k += 1
            if k == 1:
                Z = rk / v
                p = np.copy(Z)
                rho_km1 = (rk.transpose()).dot(Z)
            else:
                beta = rho_km1 / rho_km2
                p = Z + beta * p

            # Update search direction efficiently
            w = x * A.dot(x * p) + v * p
            alpha = rho_km1 / (((p.transpose()).dot(w))[0, 0])
            ap = alpha * p
            # Test distance to boundary of cone
            ynew = y + ap

            if np.amin(ynew) <= delta:
                if delta == 0:
                    break
                ind = np.where(ap < 0.0)[0]
                gamma = np.amin((delta - y[ind]) / ap[ind])
                y += gamma * ap
                break

            if np.amax(ynew) >= Delta:
                ind = np.where(ynew > Delta)[0]
                gamma = np.amin((Delta - y[ind]) / ap[ind])
                y += gamma * ap
                break

            y = np.copy(ynew)
            rk -= alpha * w
            rho_km2 = rho_km1
            Z = rk / v
            rho_km1 = ((rk.transpose()).dot(Z))[0, 0]

        x *= y
        v = x * (A.dot(x))
        rk = 1.0 - v
        rho_km1 = ((rk.transpose()).dot(rk))[0, 0]
        rout = rho_km1
        MVP += k + 1

        # Update inner iteration stopping criterion
        rat = rout / rold
        rold = rout
        res_norm = rout ** 0.5
        eta_o = eta
        eta = g * rat
        if g * eta_o ** 2.0 > 0.1:
            eta = max(eta, g * eta_o ** 2.0)
        eta = max(min(eta, etamax), stop_tol / res_norm)

    x = sps.diags(x.flatten(), 0, format='csr')
    Ahat = x.dot(A.dot(x))
    return Ahat


def clique_expand_incidence(I, zero_diag=True):
    """Performs clique expansion on an incidence matrix.

    This function takes an incidence matrix and identifies potential 
    larger cliques based on the overlap of nodes within smaller cliques.

    Args:
        I (pd.DataFrame): An incidence matrix where rows represent nodes 
                          and columns represent edges.
        zero_diag (bool, optional): If True, sets the diagonal entries of 
                                    the result matrix to zero. Defaults to True.

    Returns:
        pd.DataFrame: A matrix where non-zero entries indicate nodes that 
                      potentially form a larger clique.
    """

    node_list = I.index
    A = np.dot(I, I.T)
    if zero_diag:
        A = A - np.diag(np.diag(A))  
    A = pd.DataFrame(A, columns=node_list, index=node_list)
    return A


def symmetrize(arr, method="average"):
    """
    Symmetrizes a square matrix.

    Args:
        arr (np.ndarray): The input square matrix.
        method (str, optional): The method used for symmetrization. 
            * "average": Averages the upper and lower triangular parts. (Default)
            * "upper": Reflects the upper triangular part to the lower part.
            * "lower": Reflects the lower triangular part to the upper part.

    Returns:
        np.ndarray: The symmetrized matrix.
    """

    if arr.shape[0] != arr.shape[1]:
        raise ValueError("Input array must be square.")

    if method == "average":
        return (arr + arr.T) / 2
    elif method == "upper":
        return arr + np.tril(arr, k=-1).T 
    elif method == "lower":
        return arr + np.triu(arr, k=1).T
    else:
        raise ValueError("Invalid method. Choose from 'average', 'upper', or 'lower'")