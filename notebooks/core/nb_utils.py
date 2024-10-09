import pandas as pd
import numpy as np
from collections import Counter
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.linalg import toeplitz
import scipy.sparse as sps
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import issparse
import cooler

import os
import sys
import subprocess


def nested_list_to_incidence_matrix(data):
    """Converts a list of edges to a corresponding incidence matrix.

    Args:
        data: A list of lists, where each sublist represents edges connected
              to a common node (the first element of the sublist).

    Returns:
        A NumPy array representing the incidence matrix.
    """

    # Find all unique nodes efficiently using set comprehension
    all_nodes = {node for sublist in data for node in sublist}
    num_nodes = len(all_nodes)

    # Create a sparse matrix for better memory usage with many nodes/edges
    incidence_matrix = np.zeros((num_nodes, len(data)), dtype=int)

    # Iterate through edges, converting nodes to indices using a dictionary
    node_to_index = {node: i for i, node in enumerate(all_nodes)}
    for i, sublist in enumerate(data):
        for node in sublist:  # Skip the first node (already processed)
            incidence_matrix[node_to_index[node], i] = 1

    return incidence_matrix


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
    

def normalize_1d_array(arr, target_min=0, target_max=1):
    """Normalizes a 1D NumPy array to a specified range.

    Args:
        arr (np.ndarray): The 1D NumPy array to normalize.
        target_min (float, optional): The desired minimum value in the normalized range. Defaults to 0.
        target_max (float, optional): The desired maximum value in the normalized range. Defaults to 1.

    Returns:
        np.ndarray: The normalized array with values within the specified range.
    """

    arr_min = arr.min()
    arr_max = arr.max()

    normalized = (arr - arr_min) / (arr_max - arr_min)  # Normalize to [0, 1]
    normalized = normalized * (target_max - target_min) + target_min  # Scale to the target range

    return normalized
    

def identify_off_diagonal_outliers(matrix, threshold=3.0):
    """Identifies indices of off-diagonal outliers in a symmetric matrix.

    Outliers are determined based on a z-score threshold applied to the absolute values
    of the off-diagonal elements.

    Args:
        matrix (np.ndarray): The input symmetric matrix.
        threshold (float, optional): The z-score threshold for outlier detection. 
                                     Defaults to 3.0.

    Returns:
        list: A list of tuples where each tuple represents an outlier index 
              pair (row, column).
    """

    if not np.allclose(matrix, matrix.T):
        raise ValueError("Input matrix is not symmetric.")

    # Extract off-diagonal elements, excluding the main diagonal
    off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]

    # Calculate z-scores based on absolute values
    abs_off_diagonal = np.abs(off_diagonal) 
    z_scores = (abs_off_diagonal - abs_off_diagonal.mean()) / abs_off_diagonal.std()

    # Identify outliers based on threshold
    outlier_indices = np.where(z_scores > threshold) 

    # Create a list of (row, column) index pairs
    outlier_pairs = []
    n = matrix.shape[0]
    for idx in outlier_indices[0]:  
        row = idx // n  
        col = idx % n  
        if row != col:  # Ensure we only include off-diagonal outliers 
            outlier_pairs.append((row, col))

    return outlier_pairs 
    

def explicit_clique_expand(I):
    """Performs explicit clique expansion on an incidence matrix.

    This function identifies pairs of nodes that co-occur within the same edges
    and builds a matrix representing potential higher-order cliques.

    Args:
        I (pd.DataFrame): An incidence matrix where rows represent nodes 
                          and columns represent edges.

    Returns:
        np.ndarray: A matrix where non-zero entries indicate nodes with
                    potential to form a larger clique. 
    """

    n, h = I.shape
    node_list = I.index
    clique_matrix = np.zeros((n, n))

    for edge_name in I.columns:  # Iterate over edges
        node_indices = np.argwhere(I[edge_name] > 0).ravel()
        pairs = itertools.combinations(node_indices, 2)

        for i, j in pairs:
            clique_matrix[i, j] += 1
            clique_matrix[j, i] += 1  # Ensure symmetry

    return clique_matrix 
    

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
    

def drop_duplicate_columns(df):
    """Drops duplicate columns in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to modify.

    Returns:
        pd.DataFrame: The DataFrame with duplicate columns removed.
    """
    duplicate_columns = df.columns[df.columns.duplicated(keep=False)]  
    return df.drop(duplicate_columns, axis=1) 
    

def sort_by_lowest_index(df):
    """Sorts DataFrame columns by the lowest non-zero index.

    Columns consisting entirely of zeros are placed at the end of the sorted DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to be sorted.

    Returns:
        pd.DataFrame: A new DataFrame with the columns sorted.
    """

    def key_function(column):
        """Determines the sorting key for a column based on its lowest non-zero index."""
        nonzero_indices = df[column].to_numpy().nonzero()[0]  
        return nonzero_indices.min() if len(nonzero_indices) > 0 else float('inf')  

    new_order = sorted(df.columns, key=key_function)
    return df[new_order]


def fill_missing_bins(df, bins):
    """Fills missing bins with zeros in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing bin counts.
        bins (list): A list of bin indices to include in the output.

    Returns:
        pd.DataFrame: The DataFrame with missing bins filled.
    """

    missing_bins = set(bins) - set(df.index)  # Find missing bins efficiently
    if missing_bins:
        new_index = bins  # Use the provided list of bins as the new index
        df_filled = df.reindex(new_index, fill_value=0)
    else:
        df_filled = df.copy()

    return df_filled


def process_chromosome_data(group, order_threshold, sample_size):
    """Processes data for a single chromosome with filtering, sampling, and read mapping.

    Args:
        group (pd.DataFrame): DataFrame containing contact data.
        order_threshold (int): Minimum contact order to retain.
        sample_size (int): Number of reads to sample.

    Returns:
        tuple: 
            - np.ndarray: The sampled incidence matrix.
            - dict:  Mapping from read_code to read_name.
    """

    # Calculate contact orders
    group['order'] = group.groupby('read_name')['bin'].transform('nunique')

    # Filter low-order contacts
    group = group[group['order'] > order_threshold].reset_index(drop=True)

    # Prepare for incidence matrix construction
    group['read_code'] = group['read_name'].astype('category').cat.codes
    sorted_read_codes = group['read_code'].unique()

    # Adjust sample size if necessary
    if sample_size is None:
        sample_size = len(sorted_read_codes)
    else:
        sample_size = min(sample_size, len(sorted_read_codes))  

    # Randomly sample reads
    sample_ind = np.random.choice(sorted_read_codes, sample_size, replace=False)  

    # Create read_code to read_name mapping
    read_code_map = dict(zip(group['read_code'], group['read_name']))

    # Construct the incidence matrix with sampling
    val = np.ones(len(group))
    incidence_matrix = incidence_by_pivot(group, 'read_code', 'bin', val)

    return incidence_matrix[sample_ind], read_code_map
    

def get_oht(matrix):
    """
    Computes the optimal hard threshold (OHT) index and the corresponding threshold value.
    This implementation assumes an unknown noise level and uses a heuristic formula based on matrix dimensions.

    Args:
        matrix (np.ndarray): The input rectangular matrix.

    Returns:
        tuple: A tuple containing:
            * oht (int): The optimal hard threshold index for the singular values.
            * tau (float): The calculated threshold value.
    """

    m, n = matrix.shape
    beta = m / n  # Aspect ratio

    # Calculate singular values
    _, s, _ = np.linalg.svd(matrix, full_matrices=False)

    # Estimate threshold using heuristic formula
    med_y = np.median(s)
    wb = (0.56 * beta ** 3) - (0.95 * beta ** 2) + (1.82 * beta) + 1.43
    tau = wb * med_y

    # Find indices of singular values above the threshold
    index = np.argwhere(s >= tau) 
    oht = np.max(index) if index.size > 0 else 0  # Handle potential case of no values above threshold

    return oht, tau
    

def plot_approximation_comparison(I, r=1, quantile=0.99, cmap='viridis', title=''):
    """
    Plots a comparison of a matrix, its quantile-thresholded approximation, and the difference. 
    Calculates the rank-r approximation using svd_rank_r_approx_auto.

    Args:
        I (np.ndarray): The original matrix.
        r (int, optional): The rank of the approximation. Used in the title if provided. 
        quantile (float, optional): The quantile used for thresholding.  Defaults to 0.99.
        cmap (str, optional): Colormap to use. Defaults to 'viridis'. 
        title (str, optional): The title for the entire set of plots. Defaults to ''.

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    Ihat, _, _ = svd_rank_r_approx_auto(I, r=r)
    Ihat_bin = threshold_matrix_by_quantile(Ihat, quantile)
    
    fig, axs = plt.subplots(1, 4)  

    # --- Plot 1: Original Data ---
    sns.heatmap(I, cmap=cmap, cbar=False, ax=axs[0])
    axs[0].set_yticks([])
    axs[0].set_xticks([])
    axs[0].set_ylabel("")
    axs[0].set_xlabel("")
    axs[0].set_title("Data")

    # --- Plot 2: Low-rank Approximation ---
    sns.heatmap(Ihat, cmap=cmap, cbar=False, ax=axs[1])
    axs[1].set_yticks([])
    axs[1].set_xticks([])
    axs[1].set_ylabel("")
    axs[1].set_xlabel("")
    axs[1].set_title(f"Rank {r} Approx")  


    # --- Plot 3: Thresholded Approximation ---
    Ihat_bin = threshold_matrix_by_quantile(Ihat, quantile) 
    sns.heatmap(Ihat_bin, cmap=cmap, cbar=False, ax=axs[2])
    axs[2].set_yticks([])
    axs[2].set_xticks([])
    axs[2].set_ylabel("")
    axs[2].set_xlabel("")
    axs[2].set_title(f"Thresholded Approx {quantile=}")  

    # --- Plot 4: Difference ---
    D = np.abs(I - Ihat_bin)
    sns.heatmap(D, cmap=cmap, cbar=False, ax=axs[3])
    axs[3].set_yticks([])
    axs[3].set_xticks([])
    axs[3].set_ylabel("")
    axs[3].set_xlabel("")
    axs[3].set_title("Difference")
    
    plt.suptitle(title)
    plt.tight_layout()  # Adjust spacing between subplots 
    return fig
    

def svd_rank_r_approx_auto(matrix, r):
    """Calculates a rank-r approximation of a matrix using Singular Value Decomposition (SVD).

    This function performs Truncated SVD, which decomposes the input matrix into a 
    product of three matrices (U, Sigma, V^T). It then reconstructs a low-rank
    approximation using only the top 'r' singular values and corresponding vectors.

    Args:
        matrix (np.ndarray or scipy.sparse matrix): The input matrix to approximate.
        r (int): The desired rank of the approximation. Must be less than or 
                 equal to the minimum dimension of the input matrix.

    Returns:
        tuple: A tuple of three elements:
            * np.ndarray or scipy.sparse.csr_matrix: The rank-r approximated matrix (same type and shape as input).
            * np.ndarray: The embedded representation of the matrix in the reduced r-dimensional space.
            * TruncatedSVD: The fitted TruncatedSVD object.

    Raises:
        ValueError: If 'r' is greater than the minimum dimension of the input matrix.
    """

    if r > min(matrix.shape):
        raise ValueError("The desired rank 'r' must be less than or equal to the minimum dimension of the input matrix.")

    svd = TruncatedSVD(n_components=r)
    svd.fit(matrix)
    embedded = svd.transform(matrix)
    low_rank_approx = svd.inverse_transform(embedded)
    return low_rank_approx, embedded, svd 


def remove_indices(matrix, indices_to_remove):
    """Removes specified rows and columns from a matrix.

    Args:
        matrix (np.ndarray): The input matrix.
        indices_to_remove (list): A list of indices to remove.

    Returns:
        np.ndarray: The new matrix with specified rows and columns removed.
    """
    # Efficiently remove rows and columns, maintaining symmetry
    new_matrix = np.delete(np.delete(matrix, indices_to_remove, axis=0), 
                           indices_to_remove, axis=1)

    return new_matrix


def threshold_matrix_by_quantile(matrix, quantile):
    """
    Thresholds a rectangular matrix based on a specified quantile.

    Args:
        matrix (np.ndarray): The rectangular matrix to threshold.
        quantile (float): The quantile value (between 0 and 1) to use as the threshold.

    Returns:
        np.ndarray: The thresholded matrix.
    """

    if not 0 <= quantile <= 1:
        raise ValueError("Quantile must be between 0 and 1.")

    # Flatten the matrix to efficiently calculate the quantile value
    flat_matrix = matrix.flatten()

    # Calculate the threshold
    threshold = np.quantile(flat_matrix, quantile)

    # Apply thresholding using vectorized comparison 
    return np.where(matrix > threshold, 1, 0)


def drop_zero_sum(matrix):
  """
  Drops rows and columns from a matrix where the row or column sum is zero.

  Args:
      matrix (np.ndarray): A 2D NumPy array.

  Returns:
      np.ndarray: The matrix with zero-sum rows and columns removed.
  """

  # Calculate row and column sums
  row_sums = np.sum(matrix, axis=1)
  col_sums = np.sum(matrix, axis=0)

  # Find indices of rows and columns to keep 
  keep_rows = row_sums != 0
  keep_cols = col_sums != 0

  # Return the sub-matrix 
  return matrix[keep_rows][:, keep_cols]
    

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
        

def drop_below_diag_threshold(arr, threshold):
    """
    Drops rows and columns of a square, symmetric NumPy array where the diagonal is below a threshold.

    Args:
        arr (np.ndarray): A square, symmetric NumPy array.
        threshold (float): The threshold value for the diagonal elements.

    Returns:
        np.ndarray: The modified array with rows and columns dropped.
    """

    if not np.allclose(arr, arr.T):
        raise ValueError("Input array must be symmetric.")

    if arr.shape[0] != arr.shape[1]:
        raise ValueError("Input array must be square.")

    # Find indices of rows/columns to keep (where diagonal is above threshold)
    keep_mask = np.diag(arr) >= threshold

    # Return the sub-array with rows and columns masked by keep_mask
    return arr[keep_mask][:, keep_mask]


def incidence_by_pivot(df, index, columns, values):
  """
  A function to make an incidence matrix through a pivot table.

  Args:
      df (pd.DataFrame): Input DataFrame containing the data.
      index (str): Column name to be used as the index in the pivot table.
      columns (list): List of column names to be used as columns in the pivot table.
      values (str or list): 
          - If a string, it's assumed to be a column name representing values 
            used to calculate incidence.
          - If a list, it should be the same length as the number of unique values 
            in the `index` column used to create the new index during pivoting.

  Returns:
      pd.DataFrame: The incidence matrix as a transposed pivot table.
  """

  # Check if values is a string (column name) or a list
  if isinstance(values, str):
    value_col = values
    rm = False
  else:
    rm = True
    value_col = f"temp_col_{len(df)}"  # Create a unique temporary column name
    df[value_col] = values

  # Create the pivot table with the temporary column (if needed)
  I = pd.pivot_table(df, index=index, columns=columns, values=value_col, fill_value=0)

  # Drop the temporary column if it was created
  if rm:
    del df[value_col]

  # Return the transposed incidence matrix
  return I.T
    

def bin_list_to_incidence(df, bins, reads):
    """
    Efficiently one-hot encode bins for large DataFrames.
    
    Args:
      df (pd.DataFrame): Input DataFrame containing data.
      bins (str): Column containing lists of bin values.
      reads (str): Column containing unique read codes.
    
    Returns:
      pd.DataFrame: One-hot encoded DataFrame with:
          - Rows indexed by read codes.
          - Columns for one-hot encoded bins (integer names).
          - Missing values filled with zeros.
    
    Optimizes memory for large DataFrames.
    """

    def g(df):
        # Explode bins column within each group
        df_expanded = df.explode(bins)
        df_expanded[bins] = df_expanded[bins].astype(str)

        # One-hot encode the bins
        one_hot = pd.get_dummies(df_expanded[bins])

        # Merge with original data and drop bins column
        result = pd.concat([df_expanded, one_hot], axis=1).drop(columns=[bins])
        return result

    # Apply one-hot encoding in chunks for memory efficiency
    result = df.groupby(reads).apply(g).reset_index(drop=True)
    result = result.set_index(reads)
    # Fill NaN values with zeros
    result.fillna(0, inplace=True)

    # Change column index datatype to integer
    result.columns = result.columns.astype(int)
    return result.sort_index(axis=1)


def human_readable_bp(bp, base=1000, suffix="b"):
    """
    This function translates a number of bp into a human-readable string format (e.g., KB, MB, GB, TB).
    
    Args:
      bp: The number of bases to convert (int).
      base: the base number, for example: 1024 for bytes
    
    Returns:
      A human-readable string representation of the size (str).
    """
    
    suffixes = ["", "K", "M", "G", "T", "P", "E"]
    suffixes = [f"{x}{suffix}" for x in suffixes]
    i = 0
    while bp >= base and i < len(suffixes) - 1:
        bp /= base
        i += 1
    return f"{int(bp)}{suffixes[i]}"



def loadDNFeatures(paths, chroms, 
                   chrom="2", resolution=1000000):
    """A function to load 1d features"""
    
    chromLen = chroms[chroms['chrom'] == chrom]['size'].values[0]
    nBins = np.ceil(chromLen / resolution).astype(int)
    dnData = {}
    for track in os.listdir(paths):
        trackDir = f"{paths}{track}/"
        fCount = 0
        for tFiles in os.listdir(trackDir):
            if ".bw" in tFiles:
                fCount += 1
                trackName = f"{track}_{fCount}"
                fullPath = f"{trackDir}{tFiles}"
                bw = pyBigWig.open(fullPath)
                vec = bw.stats(f"chr{chrom}", type="mean", nBins=nBins)
                dnData[trackName] = vec
                
    gf = pd.DataFrame.from_dict(dnData, orient='index').transpose()
    gf = gf.fillna(0.0)
    return gf


def vec2color(vector, cmap='viridis'):
    """A function to map a cell id to a color
    from a color map """
    colors = {}
    cmap = get_cmap(cmap)
    
    for i, item in enumerate(vector):
        color = cmap(i / len(vector))
        colors[item] = color
    return colors


def get_bintervals(sizes, chrom, resolution):
    """A cheeky name. returns the start/end position of 
    each bin for a given chromosome """
    chrom_len = sizes.loc[sizes['chrom'] == chrom]['size'].values[0]
    n_bins = int(np.ceil(chrom_len / resolution)) + 1 # to account for partial bins

    bdf = pd.DataFrame({'Bin' : list(range(n_bins))})
    bdf['Start'] = bdf['Bin'] * resolution
    bdf['End'] = bdf['Start'] + (resolution - 1)
    bdf.insert(0, 'Chromosome', [chrom]*n_bins)
    return bdf


def bin_loci(position, bin_size=1000000):
    """
    Convert a genomic position to the corresponding bin index.

    Args:
    - position (int): The genomic position.
    - bin_size (int): The size of each bin in base pairs. Default is 1 Mb (1000000).

    Returns:
    - bin_index (int): The index of the bin that the position falls into.
    """
    bin_index = np.ceil(position / bin_size)
    return bin_index

    

def tail(f, lines=20):
    total_lines_wanted = lines

    BLOCK_SIZE = 1024
    f.seek(0, 2)
    block_end_byte = f.tell()
    lines_to_go = total_lines_wanted
    block_number = -1
    blocks = []
    while lines_to_go > 0 and block_end_byte > 0:
        if (block_end_byte - BLOCK_SIZE > 0):
            f.seek(block_number*BLOCK_SIZE, 2)
            blocks.append(f.read(BLOCK_SIZE))
        else:
            f.seek(0,0)
            blocks.append(f.read(block_end_byte))
        lines_found = blocks[-1].count(b'\n')
        lines_to_go -= lines_found
        block_end_byte -= BLOCK_SIZE
        block_number -= 1
    all_read_text = b''.join(reversed(blocks))
    return b'\n'.join(all_read_text.splitlines()[-total_lines_wanted:])


def read_pairs(fpath):
    """A function to read a .pairs.gz file """

    command = [
        "zgrep",
        "'#columns'",
        fpath
    ]
    output = subprocess.run(" ".join(command), shell=True, capture_output=True)
    columns = output.stdout.decode('utf-8').strip().split(" ")
    # drop the first item
    columns = columns[1:]

    df = pd.read_csv(fpath, 
                     sep='\t', 
                     header=None, 
                     names=columns,
                     comment="#",
                    )
    return df


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


def load_seqkit_report(fpath):
    """A function to load a seqkit summary """
    pdf = pd.read_csv(fpath, sep=r"\s+")
    
    columns = [
        'num_seqs',
        'sum_len',
        'min_len',
        'avg_len',
        'max_len',
        'Q1',
        'Q2',
        'Q3',
        'N50',
    ] 
    
    for c in columns:
        pdf[c] = pdf[c].astype(str).str.replace(',', '').astype(float)

    return pdf
    
    

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


def min_max(values):
    """Scales a series of values to the range [0, 1] using NumPy.

    Args:
        values: A Pandas Series or a NumPy array of numeric values.

    Returns:
        The scaled values as a Pandas Series or a NumPy array.
    """

    range_val = np.ptp(values)  # Peak-to-peak (max - min)
    if range_val == 0:
        return values
    
    scaled_values = (values - values.min()) / range_val
    return scaled_values