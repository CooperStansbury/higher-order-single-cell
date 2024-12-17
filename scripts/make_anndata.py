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
        'Overlap': 'overlap',
    }

    df = df.rename(columns=result_columns)

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
  print(f"{X.shape=}")
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

  var = var.groupby(['basename', 'read_name', 'read_index']).agg(
      mean_mapq=('mapping_quality', 'mean'),
      median_mapq=('mapping_quality', 'median'),
      n_chromosomes=('chrom', 'nunique'),
      order=('order', 'first'),
      n_bins=('bin', 'nunique'),
      read_length_bp=('length_on_read', 'sum'),
  ).reset_index()

  # Ensure proper sorting using var_names
  var = var.set_index('read_index')
  var = var.reindex(var_names)
  var = var.reset_index()
  var = var.set_index('read_name')

  return var


def create_obs_df(df, X):
  """Creates an observation DataFrame from a DataFrame and a sparse matrix.

  Args:
    df (pandas.DataFrame): DataFrame with genomic data and bin information.
    X (scipy.sparse.csr_matrix): Sparse matrix representation of the data.

  Returns:
    pandas.DataFrame: DataFrame containing observation information.
  """
  obs_columns = [
      'chrom',
      'bin_start',
      'bin_end',
      'bin',
      'bin_name',
      'bin_index',
      'chrom_bin',
  ]
  obs = df[obs_columns].drop_duplicates()
  print(f"{obs.shape=}")

  # Add a column with the sum of reads per bin
  obs['n_reads'] = X.sum(axis=1)

  print(f"{obs.shape=}")
  return obs


def load_and_process_genes(fpath, obs):
  """Loads gene data, joins it with observation data, and processes the result.

  Args:
    fpath (str): Path to the gene data file (Parquet format).
    obs (pandas.DataFrame): DataFrame with observation data ('chrom', 'bin_start', 'bin_end').

  Returns:
    pandas.DataFrame: Processed DataFrame containing gene information and overlap with bins.
  """
  gdf = pd.read_parquet(fpath)
  print(f"{gdf.shape=}")

  gdf_pr = pr.PyRanges(gdf)
  obs_pr = pr.PyRanges(obs.rename(columns={
      'chrom': 'Chromosome',
      'bin_start': 'Start',
      'bin_end': 'End',
  }))

  gdf = gdf_pr.join(
      obs_pr,
      strandedness=None,
      how='left',
      report_overlap=True,

  ).df.rename(columns={
      'Chromosome': 'chrom',
      'Start': 'bin_start',
      'End': 'bin_end',
      'Start_b': 'gene_start',
      'End_b': 'gene_end',
      'length': 'gene_length',
      'Overlap': 'overlap',
  })

  # Drop unalignable genes
  gdf = gdf[gdf['bin'] >= 0].reset_index()

  keep_columns = [
      'gene_name',
      'gene_id',
      'gene_biotype',
      'chrom',
      'gene_length',
      'is_tf',
      'gene_start',
      'gene_end',
      'bin',
      'bin_name',
      'chrom_bin',
      'overlap',
  ]

  gdf = gdf[keep_columns]
  gdf['n_bins_spanned'] = gdf.groupby('gene_name')['bin'].transform('nunique')
  gdf['is_tf'] = gdf['is_tf'].astype(bool)
  gdf['is_pt_gene'] = (gdf['gene_biotype'] == 'protein_coding')
  print(f"{gdf.shape=}")

  return gdf


def aggregate_gene_info(gdf, obs, obs_names):
    """Aggregates gene information for each observation (bin).

    Args:
        gdf (pandas.DataFrame): DataFrame with gene information and bin assignments.
        obs (pandas.DataFrame): DataFrame with observation data.
        obs_names (pandas.Index): Index of unique bin names.

    Returns:
        pandas.DataFrame: Observation DataFrame with aggregated gene information.
    """
    gene_list = lambda x: ";".join(list(x))

    # Aggregate gene information for each bin
    obs_genes = gdf.groupby('bin').agg(
        n_genes=('gene_name', 'nunique'),
        n_tfs=('is_tf', 'sum'),
        n_pt_genes=('is_pt_gene', 'sum'),
        total_gene_bp=('overlap', 'sum'),
        genes=('gene_name', gene_list),
    ).reset_index()
    print(f"{obs_genes.shape=}")

    # Filter out bins not seen in Pore-C data
    obs_genes = obs_genes[obs_genes['bin'].isin(obs['bin'].values)]
    print(f"{obs_genes.shape=}")

    # Merge aggregated gene data with the observation DataFrame
    obs = pd.merge(
        obs,
        obs_genes,
        how='left',
        left_on='bin',
        right_on='bin',
    )

    # Fill missing values and convert to integer
    obs['n_genes'] = obs['n_genes'].fillna(0.0).astype(int)
    obs['n_tfs'] = obs['n_tfs'].fillna(0.0).astype(int)
    obs['total_len_bp'] = obs['total_gene_bp'].fillna(0.0).astype(int)

    # Ensure proper sorting using obs_names
    obs = obs.set_index('bin_index')
    obs = obs.reindex(obs_names)
    obs = obs.reset_index()
    obs = obs.set_index('bin_name')

    return obs


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
    print_section_header("Merging Gene Information")
    
    

    
    
    
    # # Add the interval information
    # print_section_header("Joining Intervals with Pore-C Data")
    # df = join_intervals_pyranges(df, intervals)
    # print_data_shape("Pore-C data (after join)", df.shape)
    # print_memory_usage("Join intervals")

    # # Create the AnnData objects
    # print_section_header("Creating Sparse Matrix X")
    # X, obs_names, var_names = create_X(df)
    # X_csr = csr_matrix(X)  # Create the sparse matrix
    # sparsity = 1 - X_csr.count_nonzero() / (X_csr.shape[0] * X_csr.shape[1])
    # print_data_shape("Sparse matrix X", X_csr.shape)
    # print_sparsity(sparsity)
    # print_memory_usage("Build X")

    # print_section_header("Creating Variable DataFrame (var)")
    # var = create_var_df(df, var_names)
    # print_data_shape("Variable DataFrame", var.shape)
    # print_memory_usage("Build var")

    # # Create the observation DataFrame
    # print_section_header("Creating Observation DataFrame (obs)")
    # obs = create_obs_df(df, X)
    # print_data_shape("Observation DataFrame", obs.shape)
    # print_memory_usage("Build obs")

    # print_section_header("Loading and Processing Gene Data")
    # gdf = load_and_process_genes(gene_path, obs)
    # print_data_shape("Gene DataFrame", gdf.shape)
    # print_memory_usage("Build gdf")

    # print_section_header("Aggregating Gene Information")
    # obs = aggregate_gene_info(gdf, obs, obs_names)
    # print_data_shape("Observation DataFrame (after merge)", obs.shape)
    # print_memory_usage("Merge Gene Information")

    # # Build the AnnData object
    # print_section_header("Building AnnData Object")
    # adata = an.AnnData(X=X_csr, obs=obs, var=var)  # Use the sparse matrix

    # adata.uns['genes'] = gdf.copy()
    # adata.uns['intervals'] = intervals.copy()
    # adata.uns['base_resolution'] = resolution
    # adata.uns['chrom_sizes'] = chrom
    # adata.layers["H"] = csr_matrix(adata.X.copy())
    # print_memory_usage("Make AnnData")
    # sc.logging.print_memory_usage()

    # print_section_header("Saving AnnData Object")
    # adata.write(outpath)

    # print_section_header("Pore-C Data Processing Complete!")