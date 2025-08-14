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
    outpath = sys.argv[2]

    start_time = time.time()  # Record the start time
    adata = sc.read_h5ad(anndata_path)
    end_time = time.time()  # Record the end time
    print(f"Time taken to read the file: {end_time - start_time:.2f} seconds")
    sc.logging.print_memory_usage()


    
