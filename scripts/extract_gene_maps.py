import os
import sys
import pandas as pd
import numpy as np
import glob
import time
import scipy
from scipy.sparse import csr_matrix
import anndata as an
import scanpy as sc



if __name__ == "__main__":
    anndata_path = sys.argv[1]
    genemap_outpath = sys.argv[2]
    gdf_outpath = sys.argv[3]
    anndata_outpath = sys.argv[4]

    start_time = time.time()  # Record the start time
    adata = sc.read_h5ad(anndata_path)
    end_time = time.time()  # Record the end time
    print(f"Time taken to read the file: {end_time - start_time:.2f} seconds")
    sc.logging.print_memory_usage()

    # set the x attribute
    adata.X = csr_matrix(adata.layers['H'].copy())

    # save the gene maps
    adata.uns['gene_map'].to_parquet(genemap_outpath, index=False)
    adata.uns['gdf'].to_parquet(gdf_outpath, index=False)

    # remove uneeded attributes
    del adata.uns['gdf']
    del adata.uns['gene_map']
    del adata.uns['intervals']
    del adata.layers['H']
    
    sc.logging.print_memory_usage()

    adata.write(anndata_outpath)

    
