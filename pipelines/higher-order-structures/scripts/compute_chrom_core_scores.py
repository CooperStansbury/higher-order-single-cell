import os
import sys
import pandas as pd
import numpy as np
import glob
import time
import gget
import scipy
from scipy.sparse import csr_matrix
import anndata as an
import scanpy as sc
import networkx as nx

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

source_path = os.path.abspath("../source/")
sys.path.append(source_path)

source_path = os.path.abspath("source/")
sys.path.append(source_path)

import centrality as central
import matrix


def extract_chromosome(adata, chromosome, order_threshold=1, min_read_count=1):
    """
    Filters an AnnData object based on chromosome, order threshold, and read count.

    Args:
        adata: AnnData object to filter.
        chromosome: Chromosome to select.
        order_threshold: Minimum order (sum of reads across cells) for a gene to be kept.
        min_read_count: Minimum read count for a gene to be kept.

    Returns:
        Filtered AnnData object.
    """
    mask = (adata.obs['chrom'] == chromosome)
    cdata = adata[mask,].copy()
    cdata = cdata[:, cdata.X.sum(axis=0) > min_read_count]

    # recompute the order
    cdata.var['chrom_order'] = np.ravel(cdata.X.sum(axis=0))
    cdata = cdata[:, cdata.var['chrom_order'] > order_threshold]

    # recompute the reads
    cdata.obs['chrom_degree'] = np.ravel(cdata.X.sum(axis=1))

    # Sort the entire AnnData object (cdata) by 'chrom_bin'
    cdata = cdata[cdata.obs.sort_values('chrom_bin').index, :]

    return cdata.copy()


def find_outliers(df_column):
  """
  Identifies outliers in a pandas DataFrame column using the IQR method.

  Args:
    df_column: A pandas Series representing the column to analyze.

  Returns:
    A boolean mask with True for outliers and False otherwise.
  """
  Q1 = df_column.quantile(0.15)
  Q3 = df_column.quantile(0.85)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  return (df_column < lower_bound) | (df_column > upper_bound)


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


if __name__ == "__main__":
    anndata_path = sys.argv[1]
    chromosome = sys.argv[2]
    outpath = sys.argv[3]

    start_time = time.time()  # Record the start time
    adata = sc.read_h5ad(anndata_path)
    end_time = time.time()  # Record the end time
    print(f"Time taken to read the file: {end_time - start_time:.2f} seconds")
    sc.logging.print_memory_usage()

    # extract the chromosome
    adata = extract_chromosome(adata, chromosome)

    # drop outliers
    print("--- Finding outlier loci ---")  # Status message
    adata.obs['degree_outlier'] = find_outliers(adata.obs['chrom_degree'])
    remove_bins = adata.obs[adata.obs['degree_outlier']].index.to_list()
    print(f"Removing top {len(remove_bins)} outlier loci:")
    print(remove_bins)

    adata = adata[~adata.obs_names.isin(remove_bins), :].copy()

    """ Clique-expanded centralities """
    print("--- Expanding and normalizing AnnData ---")  # Status message
    matrix.expand_and_normalize_anndata(adata, oe_kr=True)

    ce_centralities = {
        'ce_eigenvector_centrality' : nx.eigenvector_centrality,
        'ce_betweenness_centrality' : nx.betweenness_centrality,
        'ce_pagerank' : nx.pagerank,
    }
    
    A = adata.obsm['A_oe'].copy()
    G = nx.from_pandas_adjacency(A)

    svd = TruncatedSVD(n_components=1, n_iter=10)
    adata.obs['ce_singular_vector_1'] = min_max(svd.fit_transform(A))
    
    for label, func in ce_centralities.items():
        print(f"--- Calculating {label} ---")  # Status message
        centrality = func(G, weight='weight')
            
        adata.obs[label] = adata.obs.index.map(centrality)
        adata.obs[label] = min_max(adata.obs[label])

    """ higher-order centralities """
    print("--- Adding principal singular value of the incidence matrix ---")  # Status message
    H = adata.to_df().copy()
    print(f"Raw: {H.shape=}")
    H = H.T.drop_duplicates().T
    print(f"De-duped: {H.shape=}")

    svd = TruncatedSVD(n_components=1, n_iter=10)
    adata.obs['hge_singular_vector_1'] = min_max(svd.fit_transform(H))

    # hypergraph centralities
    hge_functions = {
        'hge_logexp_unweighted': None,
        'hge_logexp_degree_weighted': 1 / (H.sum(axis=1).values + 1),
        'hge_logexp_RNA_weighted': 1 / (adata.obs.loc[H.index, 'RNA_2'].fillna(0.0).values + 1),
        'hge_logexp_ATAC_weighted': 1 / (adata.obs.loc[H.index, 'ATACSeq_1'].fillna(0.0).values + 1),
    }

    hge_centralities = []

    for label, weights in hge_functions.items():
        start_time = time.time()  # Record start time
        print(f"--- Calculating {label} ---")  # Status message
        node, edge = central.nonlinear_eigenvector_centrality(
            H, function="log-exp", node_weights=weights,
        )
    
        hge_centralities.append(label)
        adata.obs[label] = min_max(node)
    
        end_time = time.time()  # Record end time
        print(f"{label} calculation took: {end_time - start_time:.2f} seconds")

    print("--- Saving results ---")  # Status message
    obs = adata.obs.copy()
    obs = obs.reset_index(drop=False)
    obs.to_csv(outpath, index=False)
    print("--- Done! ---")  # Status message

    

    


    

    

    



    