import os
import sys
import pandas as pd
import numpy as np
import glob
import time
from scipy.sparse import csr_matrix
import anndata as an
import scanpy as sc
import pyranges as pr
import psutil


def print_memory_usage(step_name):
    """Prints the current RAM usage in GB."""
    process = psutil.Process()
    mem_gb = process.memory_info().rss / (1024 ** 3)  # Memory in GB
    print(f"RAM usage at step '{step_name}': {mem_gb:.2f} GB") 
    

def print_section_header(title):
    """Prints a visually appealing section header."""
    print("-" * 60)
    print(f" {title} ".center(60, "-"))
    print("-" * 60)


def print_parameter(name, value):
    """Prints a parameter with consistent formatting."""
    print(f"{name:<35} {value}")


def print_data_shape(name, shape):
    """Prints data shape information consistently."""
    print_parameter(f"{name} shape:", shape)


def print_sparsity(sparsity):
    """Prints sparsity as a percentage."""
    print_parameter("Sparsity of X:", f"{sparsity:.2%}")


def merge_genes(df, gdf):
    """Merges gene data from a Parquet file into a DataFrame.

    Args:
        df: DataFrame with genomic intervals ('chrom', 'ref_start', 'ref_end').
        gdf: DataFrame with gene data.

    Returns:
        DataFrame with added gene information.
    """
    # Use PyRanges for efficient interval joining
    gdf_pr = pr.PyRanges(gdf)
    df_pr = pr.PyRanges(df.rename(columns={
        'chrom': 'Chromosome',
        'ref_start': 'Start',
        'ref_end': 'End',
    }))

    # Join dataframes, keeping all original intervals
    df = df_pr.join(
        gdf_pr,
        strandedness=None,
        how='left',
        report_overlap=True,
    ).df.rename(columns={
        'Chromosome': 'chrom',
        'Start': 'ref_start',
        'End': 'ref_end',
        'Start_b': 'gene_start',
        'End_b': 'gene_end',
        'length': 'gene_length',
        'Overlap': 'gene_overlap',
    })

    # Select the best overlap for each interval
    df = df.sort_values(by='gene_overlap', ascending=False)
    df = df.drop_duplicates(subset=['read_name', 'read_start', 'ref_start', 'basename'], keep='first')

    # Ensure correct data types
    df['is_tf'] = df['is_tf'].astype(bool)
    df['is_pt_gene'] = (df['gene_biotype'] == 'protein_coding')
    return df


def get_intervals(bp, resolution):
    """Generates non-overlapping intervals within a given range.

    Args:
        bp (int): The upper bound of the range.
        resolution (int): The size of each interval.

    Returns:
        numpy.ndarray: An array of intervals, where each row is [start, end].
    """
    edges = np.arange(0, bp + resolution, resolution)
    return np.column_stack((edges[:-1], edges[1:]))


def create_bin_table(chroms, resolution):
    """Creates a table of genomic bins for given chromosomes and resolution.

    Args:
        chroms (pandas.DataFrame): DataFrame with 'chrom' and 'size' columns.
        resolution (int): The size of each bin.

    Returns:
        pandas.DataFrame: A DataFrame with 'chrom', 'start', and 'end' columns for each bin.
    """
    bin_table = []
    for _, row in chroms.iterrows():
        chrom, bp = row
        bins = get_intervals(bp, resolution)
        chr_df = pd.DataFrame(bins, columns=['start', 'end'])
        chr_df['chrom'] = chrom
        bin_table.append(chr_df)
    return pd.concat(bin_table, ignore_index=True)


def create_chromosome_intervals(fpath, base_resolution=10000):
    """Creates a dataframe of chromosome intervals.

    Args:
        fpath (str): Path to the CSV file with chromosome sizes ('chrom', 'size' columns).
        base_resolution (int): The desired resolution for the intervals.

    Returns:
        tuple: 
            - pandas.DataFrame: Original chromosome sizes.
            - pandas.DataFrame: DataFrame with 'chrom', 'start', 'end', 'bin', 'chrom_bin', and 'bin_name' columns.
    """
    chrom = pd.read_csv(fpath)
    intervals = create_bin_table(chrom[['chrom', 'size']], base_resolution)
    intervals = intervals.reset_index(drop=False, names='bin')
    intervals = intervals[['chrom', 'start', 'end', 'bin']]
    intervals['chrom_bin'] = intervals.groupby('chrom')['bin'].cumcount()

    intervals['start'] = intervals['start'].astype(int)
    intervals['end'] = intervals['end'].astype(int)
    intervals['bin_name'] = "chr" + intervals['chrom'] + ":" + intervals['chrom_bin'].astype(str)
    print(f"{intervals.shape=}")
    return chrom, intervals


def join_intervals_pyranges(df, intervals):
    """Joins a DataFrame with genomic intervals using PyRanges.

    Args:
        df (pandas.DataFrame): DataFrame with genomic data ('chrom', 'start', 'end').
        intervals (pandas.DataFrame): DataFrame with intervals ('chrom', 'start', 'end').

    Returns:
        pandas.DataFrame: The joined DataFrame with overlap information and indices.
    """

    pyranges_columns = {
        'chrom': 'Chromosome',
        'start': 'Start',
        'end': 'End',
        'ref_start': 'Start',
        'ref_end': 'End',
    }

    # Rename columns for PyRanges compatibility
    intervals = intervals.rename(columns=pyranges_columns)
    df = df.rename(columns=pyranges_columns)

    # Create PyRanges objects
    df_pr = pr.PyRanges(df)
    intervals_pr = pr.PyRanges(intervals)

    # Perform the join operation
    df = df_pr.join(
        intervals_pr,
        strandedness=None,
        how='left',
        report_overlap=True,
    ).df

    result_columns = {
        'Chromosome': 'chrom',
        'Start': 'ref_start',
        'End': 'ref_end',
        'Start_b': 'bin_start',
        'End_b': 'bin_end',
        'Overlap': 'bin_overlap',
    }

    df = df.rename(columns=result_columns)

    # Select the best overlap for each interval
    df = df.sort_values(by='bin_overlap', ascending=False)
    df = df.drop_duplicates(subset=['read_name', 'read_start', 'ref_start', 'basename'], keep='first')

    # Add index identifiers
    df['read_index'] = pd.factorize(df['read_name'])[0]
    df['bin_index'] = pd.factorize(df['bin'])[0]

    print(f"{df.shape=}")
    return df


def create_X(df):
  """Creates a sparse matrix and associated names from a DataFrame.

  Args:
    df (pandas.DataFrame): DataFrame with 'value', 'bin_index', and 'read_index' columns.

  Returns:
    tuple: 
        - scipy.sparse.csr_matrix: The sparse matrix.
        - numpy.ndarray: Unique bin indices (obs_names).
        - numpy.ndarray: Unique read indices (var_names).
  """
  data = df['value'].tolist()
  row = df['bin_index'].values
  col = df['read_index'].values

  n = df['bin_index'].nunique()
  m = df['read_index'].nunique()

  obs_names = df['bin_index'].unique()
  var_names = df['read_index'].unique()

  X = csr_matrix((data, (row, col)), shape=(n, m))
  X = csr_matrix((X > 0).astype(int))
  return X, obs_names, var_names


def create_var_df(df, var_names):
  """Creates a variable DataFrame from a DataFrame with read information.

  Args:
    df (pandas.DataFrame): DataFrame with read data including 'read_name', 'read_index', 
                           'mapping_quality', 'chrom', 'order', 'bin', and 'length_on_read'.
    var_names (pandas.Index): Index of unique read names.

  Returns:
    pandas.DataFrame: DataFrame containing variable information (read-level summaries).
  """
  var = df.copy()

  gene_list = lambda x: ";".join([i for i in set(x) if i != '-1'])
  n_genes = lambda x: len([i for i in set(x) if i != '-1'])

  var = var.groupby(['basename', 'read_name', 'read_index']).agg(
      mean_mapq=('mapping_quality', 'mean'),
      median_mapq=('mapping_quality', 'median'),
      n_chromosomes=('chrom', 'nunique'),
      order=('order', 'first'),
      n_bins=('bin', 'nunique'),
      read_length_bp=('length_on_read', 'sum'),
      genes=('gene_name', gene_list),
      n_genes=('gene_name', n_genes),
  ).reset_index()

  # Ensure proper sorting using var_names
  var = var.set_index('read_index')
  var = var.reindex(var_names)
  var = var.reset_index()
  var = var.set_index('read_name')

  return var


def create_obs_df(df, obs_names):
  """Create a DataFrame with observation information.

  Args:
    df: DataFrame with genomic data and bin information.
    obs_names: List of observation names to ensure proper sorting.

  Returns:
    DataFrame containing observation information.
  """
  gene_list = lambda x: ";".join([i for i in set(x) if i != '-1'])
  n_genes = lambda x: len([i for i in set(x) if i != '-1'])
    
  obs = df.groupby('bin_name').agg(
      bin_start=('bin_start', 'first'),
      bin_end=('bin_end', 'first'),
      bin=('bin', 'first'),
      bin_index=('bin_index', 'first'),
      chrom=('chrom', 'first'),
      chrom_bin=('chrom_bin', 'first'),
      degree=('read_name', 'nunique'),
      genes=('gene_name', gene_list),
      n_genes=('gene_name', n_genes),
  ).reset_index()

  # Ensure proper sorting using var_names
  obs = obs.set_index('bin_index')
  obs = obs.reindex(obs_names)
  obs = obs.reset_index()
  obs = obs.set_index('bin_name')

  return obs


def create_gene_map(df):
  """Creates a gene map DataFrame.

  Args:
    df: DataFrame with gene information.

  Returns:
    DataFrame containing the gene map with duplicates removed and '-1' gene names excluded.
  """
  gene_map = df[['gene_name', 'gene_biotype', 'read_name', 'bin_name']].drop_duplicates()
  gene_map = gene_map[gene_map['gene_name'] != '-1']
  gene_map = gene_map.reset_index(drop=True)
  return gene_map



if __name__ == "__main__":
    pore_c_path = sys.argv[1]
    resolution = int(sys.argv[2])
    chrom_path = sys.argv[3]
    gene_path = sys.argv[4]
    outpath = sys.argv[5]

    print_section_header("Pore-C Data Processing Report")

    print_parameter("Pore-C data path:", pore_c_path)
    print_parameter("Resolution:", resolution)
    print_parameter("Chromosome sizes path:", chrom_path)
    print_parameter("Gene annotations path:", gene_path)
    print_parameter("Output path:", outpath)

    print_section_header("Initialization")
    print_memory_usage("Initialization")

    # Load the Pore-C data
    print_section_header("Loading Pore-C Data")
    df = pd.read_parquet(pore_c_path)
    df['value'] = 1
    print_data_shape("Pore-C data", df.shape)
    print_memory_usage("Load Pore-C data")

    # Load the chromosome table
    print_section_header("Creating Chromosome Intervals")
    chrom, intervals = create_chromosome_intervals(chrom_path, base_resolution=resolution)
    print_data_shape("Chromosome intervals", intervals.shape)
    print_memory_usage("Create chromosome intervals")

    # map the genes at read level
    print_section_header("Merging Gene Information with Pore-C (Read-level)")
    gdf = pd.read_parquet(gene_path)
    print_data_shape("Gene data (Genes added)", gdf.shape)
    df = merge_genes(df, gdf)
    print_data_shape("Pore-C data (Genes added)", df.shape)
    print_memory_usage("Merged Genes")
    
    # Add the interval information
    print_section_header("Joining Intervals with Pore-C Data")
    df = join_intervals_pyranges(df, intervals)
    print_data_shape("Pore-C data (Bins Added)", df.shape)
    print_memory_usage("Join intervals")

    # Create the AnnData objects
    print_section_header("Creating Sparse Matrix X")
    X_csr, obs_names, var_names = create_X(df)
    sparsity = 1 - X_csr.count_nonzero() / (X_csr.shape[0] * X_csr.shape[1])
    print_data_shape("Sparse matrix X", X_csr.shape)
    print_sparsity(sparsity)
    print_memory_usage("Build X")

    print_section_header("Creating Variable DataFrame (var)")
    var = create_var_df(df, var_names)
    print_data_shape("Variable DataFrame", var.shape)
    print_memory_usage("Build var")

    # Create the observation DataFrame
    print_section_header("Creating Observation DataFrame (obs)")
    obs = create_obs_df(df, obs_names)
    print_data_shape("Observation DataFrame", obs.shape)
    print_memory_usage("Build obs")

    # Build the AnnData object
    print_section_header("Building AnnData Object")
    adata = an.AnnData(X=X_csr, obs=obs, var=var)  # Use the sparse matrix

    print_section_header("Adding Unstructured Annotations (uns)")
    adata.uns['gene_map'] = create_gene_map(df).copy()
    adata.uns['gdf'] = gdf.copy()
    adata.uns['intervals'] = intervals.copy()
    adata.uns['base_resolution'] = resolution
    adata.uns['chrom_sizes'] = chrom.copy()
    adata.layers["H"] = csr_matrix(adata.X.copy())
    print_memory_usage("Make AnnData")

    print_section_header("Saving AnnData Object")
    adata.write(outpath)

    print_section_header("Pore-C Data Processing Complete!")