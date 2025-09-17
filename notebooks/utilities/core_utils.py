import sys
import os
import pandas as pd
import numpy as np
import glob
import pyBigWig
import matplotlib.pyplot as plt
from matplotlib import patches
import seaborn as sns
import pandas as pd
import numpy as np
import random
from scipy import sparse
from scipy.stats import qmc


def generate_params(parameters, num_samples):
    """
    Generates Latin Hypercube samples for the given parameters and ranges.

    Args:
        parameters (dict): Dict of parameters
        num_samples (int): Number of samples to generate.

    Returns:
        pd.DataFrame: A DataFrame containing the generated samples, with columns named after the parameters.
    """
    parameter_names = list(parameters.keys())
    parameter_ranges = list(parameters.values())

    # Create a Latin Hypercube sampling engine
    sampler = qmc.LatinHypercube(d=len(parameter_names))

    # Generate samples in the [0, 1] range
    sample_points = sampler.random(n=num_samples)

    # Scale the samples to the actual parameter ranges
    scaled_samples = qmc.scale(
        sample_points, 
        l_bounds=[r[0] for r in parameter_ranges], 
        u_bounds=[r[1] for r in parameter_ranges])

    # Create a DataFrame from the scaled samples
    df = pd.DataFrame(scaled_samples, columns=parameter_names)
    df['key'] = list(range(len(df)))

    return df



def generate_core_periphery_hypergraph(num_core_nodes, num_periphery_nodes,
                                      edge_probability_core, edge_probability_periphery,
                                      avg_edge_size, core_periphery_probability):
    """
    This function generates a random hypergraph with a core-periphery structure and creates its binary incidence matrix.

    Args:
      num_core_nodes: Number of nodes in the core.
      num_periphery_nodes: Number of nodes in the periphery.
      edge_probability_core: Probability of an edge forming between core nodes.
      edge_probability_periphery: Probability of an edge forming between periphery nodes.
      avg_edge_size: Average number of nodes per edge.
      core_periphery_probability: Probability of an edge forming between a core node and a periphery node.

    Returns:
      A tuple containing four elements:
          * core_nodes: List of core nodes.
          * periphery_nodes: List of periphery nodes.
          * edges: List of edges.
          * incidence_matrix: Binary incidence matrix as a NumPy array.
    """
    # Define core and periphery nodes
    core_nodes = list(range(num_core_nodes))
    periphery_nodes = list(range(num_core_nodes, num_core_nodes + num_periphery_nodes))

    # Get total number of nodes
    total_nodes = len(core_nodes) + len(periphery_nodes)

    # Generate edges
    edges = []
    for _ in range(int(len(core_nodes) * edge_probability_core)):
        # Sample core nodes for an edge
        edge = random.sample(core_nodes, k=int(avg_edge_size))
        edges.append(edge)

    for _ in range(int(len(periphery_nodes) * edge_probability_periphery)):
        # Sample periphery nodes for an edge
        edge = random.sample(periphery_nodes, k=int(avg_edge_size))
        edges.append(edge)

    # Add edges between core and periphery nodes
    for _ in range(int(num_core_nodes * num_periphery_nodes * core_periphery_probability)):
        # Sample a core node and a periphery node
        core_node = random.choice(core_nodes)
        periphery_node = random.choice(periphery_nodes)
        # Create an edge with the core and periphery node
        edge = [core_node, periphery_node]
        # Optionally, you can sample additional nodes for the edge
        if avg_edge_size > 2:
            additional_nodes = random.sample(core_nodes + periphery_nodes, k=int(avg_edge_size) - 2)
            edge.extend(additional_nodes)
            
        edges.append(edge)

    # Create empty matrix
    incidence_matrix = np.zeros((total_nodes, len(edges)), dtype=int)

    # Fill the matrix with 1s for corresponding nodes in each edge
    for i, edge in enumerate(edges):
        for node in edge:
            incidence_matrix[node, i] = 1

    return core_nodes, periphery_nodes, edges, incidence_matrix



def plot_hypergraph(H, core_nodes=None):
    """
    Plots a hypergraph representation.

    Args:
        H: A pandas DataFrame representing the hypergraph incidence matrix.
        core_nodes: An optional list of node labels considered "core". 
                     These will be highlighted in the plot.
    """

    if core_nodes is None:
        core_nodes = []  # Default to an empty list if not provided

    for i, column in enumerate(H.columns):
        hyperedge = H[column][H[column] > 0]
        order = len(hyperedge)
        x_ind = np.ones(order) * (i + 1)

        c = np.where(hyperedge.index.isin(core_nodes), 'r', 'blue')

        # Plot nodes
        plt.scatter(
            x_ind, 
            hyperedge.index, 
            s=50, 
            c=c, 
            ec='k', 
            zorder=3, 
            label='Core' if c[0] == 'r' else 'Periphery')

        # Plot edges (connecting lines)
        plt.plot(x_ind, hyperedge.index, c='k')

    # Customize axes and labels
    plt.yticks(H.index, H.index + 1)  # Adjust node labels if needed
    plt.xticks([])
    plt.gca().invert_yaxis()
    plt.ylabel('Nodes')
    plt.xlabel('Hyperedges')

    # Add legend 
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')


def load_chromosome_feature(fpath: str, chrom: str, resolution: int) -> pd.DataFrame:
    """
    Converts a BigWig file to a DataFrame of summarized values.

    Args:
        fpath (str): Path to the input BigWig file.
        chrom (str): Chromosome to analyze (e.g., "chr1").
        resolution (int): Bin size for summarization (e.g., 1000 for 1kb bins).

    Returns:
        pd.DataFrame: A DataFrame containing file_id, bin_start, bin_end, and value columns.
    """

    with pyBigWig.open(fpath) as bw:
        chrom_length = bw.chroms().get(chrom)

        if not chrom_length:
            raise ValueError(f"Chromosome '{chrom}' not found in {fpath}")

        n_bins = int(np.ceil(chrom_length / resolution))

        stats = bw.stats(chrom, nBins=n_bins, type='sum', exact=True)

        # Create the bin start and end coordinates
        # Assuming half-open intervals [start, end)
        bin_starts = np.arange(0, chrom_length, resolution)
        bin_ends = np.append(bin_starts[1:], chrom_length)
        local_bin = list(range(len(bin_starts)))

        file_id = os.path.basename(fpath).replace(".bw", "")

        df = pd.DataFrame(
            {
                "file_id": file_id,
                "local_bin" : local_bin,
                "bin_start": bin_starts,
                "bin_end": bin_ends,
                "value": stats,
            },
        )

        return df


def load_chrom_sizes(fpath):
    """
    Loads chromosome size information from a tab-separated file.

    This function reads a file containing chromosome names and sizes,
    calculates the cumulative start position for each chromosome, and
    returns the data in two formats:

    1. Pandas DataFrame: containing chromosome names, sizes, and start positions
    2. Dictionary: mapping chromosome names to their start positions

    Args:
        fpath (str): Path to the tab-separated file containing chromosome information.

    Returns:
        tuple: A tuple containing:
            - Pandas DataFrame: with columns ['chrom', 'size', 'bp_start']
            - dict: mapping chromosome names to their start positions (in base pairs)
    """
    chroms = pd.read_csv(fpath)
    # chroms = chroms.head(20) # drop unplaced contigs
    # chroms['bp_start'] = chroms['size'].cumsum()
    # chroms['bp_start'] = chroms['bp_start'].shift(1).fillna(0).astype(int)
    # chrom_starts = dict(zip(chroms['chrom'].values, chroms['bp_start'].values))
    return chroms # , chrom_starts



def load_pore_c(file_list, chrom_starts, resolution=1e6, chroms=None):
    """
    Loads and processes population Pore-C data from multiple Parquet files.

    This function reads Pore-C alignment data from Parquet files in a specified directory.
    It filters, transforms, and aggregates the data to a desired resolution, keeping only reads
    mapped to specified chromosomes and with a minimum alignment order.

    Args:
        file_list (str): list of ile paths containing the Parquet files.
        chrom_starts (dict): Dictionary mapping chromosome names to their start positions.
        resolution (float, optional): Binning resolution in base pairs. Defaults to 1e6.
        chroms (list, optional): List of chromosome names to include. If None, uses all chromosomes from chrom_starts.

    Returns:
        pandas.DataFrame: Processed Pore-C data with the following columns:
            - read_name
            - align_id
            - order
            - chrom
            - local_position
            - global_bin
            - local_bin
            - basename
    """
    
    read_columns = [
        'read_name',
        'align_id',
        'chrom', 
        'ref_start', 
        'ref_end',
        'is_mapped',
    ]
    
    keep_columns = [
        'read_name', 
        'align_id', 
        'order',
        'chrom', 
        'local_position', 
        'global_bin',
        'local_bin', 
        'basename',
    ]
    
    if chroms is None:
        chroms = list(chrom_starts.keys())
    
    df = []    
    for fpath in file_list:
        basename = os.path.basename(fpath).split(".")[0]
        tmp = pd.read_parquet(fpath, columns=read_columns)

        # Filtering & Transformations
        tmp = (
            tmp[tmp['is_mapped']]
            .loc[tmp['chrom'].isin(chroms)]
            .assign(
                local_position  = lambda df: ((df['ref_end'] - df['ref_start']) // 2) + df['ref_start'],
                chrom_start     = lambda df: df['chrom'].map(chrom_starts),
                global_position = lambda df: df['chrom_start'].astype(float) + df['local_position'].astype(float),
                global_bin      = lambda df: df['global_position'].apply(lambda x: int(np.ceil(x / resolution))),
                local_bin       = lambda df: df['local_position'].apply(lambda x: int(np.ceil(x / resolution))),
                basename        = basename
            )
            .dropna(subset=['global_bin'])
            .drop_duplicates(subset=['read_name', 'global_bin'])
        )
        
        # calculate order and drop singletons efficiently
        tmp['order'] = tmp.groupby('read_name')['global_bin'].transform('nunique')
        tmp = tmp[tmp['order'] > 1]
        
        # handle single-cell
        if tmp.empty:
            continue
        
        tmp = tmp[keep_columns]
        print(basename, tmp.shape)
        df.append(tmp)
        
    df = pd.concat(df)
    return df

