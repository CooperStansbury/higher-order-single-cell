import pandas as pd
import numpy as np

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


def generate_incidence_matrix(n_rows, n_cols, order, min_count, max_count=None):
    """Generates an incidence matrix with randomly imputed new columns.

    Args:
        n_rows: The number of rows in the incidence matrix.
        n_cols: The number of new columns to generate.
        order: The approximate mean number of 1s per column.
        min_count: The minimum number of 1s allowed in a generated column.
        max_count: The max number of 1s allowed in a generated column.

    Returns:
        A pandas DataFrame representing the imputed incidence matrix.
    """

    counts = np.random.poisson(order, size=n_cols).astype(int) # ensure that we have enouch after thresholding
    counts = np.maximum(counts, min_count)  # Ensure minimum count
    
    if not max_count is None:
        counts = np.minimum(counts, max_count)

    columns = []
    for i in range(n_cols):
        col = np.zeros(n_rows, dtype=int)
        col[np.random.choice(n_rows, size=counts[i], replace=True)] = 1
        columns.append(col)

    return pd.DataFrame(columns).T 


def drop_non_unique_columns(df):
    """Efficiently drops non-unique columns from a pandas DataFrame.

    Args:
        df: A pandas DataFrame.

    Returns:
        A new pandas DataFrame with non-unique columns dropped.
    """
    
    df = df.T.drop_duplicates().T
    return df


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


def incidence_to_list_of_list(df):
    """Converts an incidence matrix (DataFrame) to a list of lists of non-zero indices.

    Args:
        df (pd.DataFrame): The binary incidence matrix.

    Returns:
        list: A list of lists where each sublist contains indices of non-zero elements in a column.
    """
    return df.apply(lambda col: col.to_numpy().nonzero()[0].tolist()).tolist()


def list_of_list_to_incidence(data):
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


def remove_low_degree_columns(incidence_df, min_degree):
    """Removes columns with low degree from a binary incidence DataFrame.

    Args:
        incidence_df (pd.DataFrame): The binary incidence matrix.
        min_degree (int): The minimum sum threshold for a column.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """

    degrees = incidence_df.sum()  
    keep_cols = degrees >= min_degree  

    return incidence_df.loc[:, keep_cols] 


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
    I = pd.pivot_table(df, 
                       index=index, 
                       columns=columns,
                       values=value_col, 
                       fill_value=0,
                      )

    # Drop the temporary column if it was created
    if rm:
        del df[value_col]

    # Return the transposed incidence matrix
    return I.T


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


def filter_and_prepare_porec_data(df, resolution, mapq=60):
    """Filters and prepares contact data for analysis.

    Args:
        df (pd.DataFrame): The raw DataFrame containing contact data.
        resolution (int): The desired genomic resolution for binning.
        mapq (int, optional): The minimum mapping quality to filter by. Defaults to 60.

    Returns:
        pd.DataFrame: The filtered and processed DataFrame.
    """
    # Filtering
    filtered_df = df[df['mapping_quality'] >= mapq].copy()  # Filter by MAPQ
    filtered_df = filtered_df[filtered_df['fragment_id'].notna()] 

    # Binning
    filtered_df['bin'] = filtered_df['ref_start'].apply(lambda x: bin_loci(x, resolution))
    return filtered_df  


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


def read_csv(file_path, name='value'):
    """Reads a CSV file with a header into a DataFrame, skipping comment lines (starting with '#').

    Args:
        file_path (str): The path to the CSV file.
        name (str, optional): The desired column name. Defaults to 'value'.

    Returns:
        pandas.DataFrame: A DataFrame containing the data from the CSV file.
    """

    return pd.read_csv(file_path, names=[name], comment='#') 