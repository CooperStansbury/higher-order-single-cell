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


if __name__ == "__main__":
    anndata_path = sys.argv[1]
    chromosome = str(sys.argv[2])
    output_path = sys.argv[3]

    adata = sc.read_h5ad(anndata_path)
    adata = extract_chromosome(adata, chromosome)

    # free up some space
    adata.X = adata.layers['H'].copy()
    del adata.layers['H']

    # remove uneeded uns
    for key in  ['gdf', 'intervals']:
        del adata.uns[key]

    # update the gene map
    gene_map = adata.uns['gene_map'].copy()
    gene_map = gene_map[gene_map['read_name'].isin(adata.var_names)].reset_index(drop=True)
    adata.uns['gene_map'] = gene_map.copy()

    adata.write(output_path)



    
    
   