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
import pyBigWig

if __name__ == "__main__":
    output_path = sys.argv[1]
    anndata_path = sys.argv[2]
    feature_paths = sys.argv[3:]

    print("-" * 30)
    print("Starting Analysis")
    print("-" * 30)

    start_time = time.time()  # Record the start time
    adata = sc.read_h5ad(anndata_path)
    end_time = time.time()  # Record the end time
    print(f"Time taken to read the file: {end_time - start_time:.2f} seconds")
    sc.logging.print_memory_usage()

    for feature_path in feature_paths:
        bw = pyBigWig.open(feature_path)
        basename = os.path.basename(feature_path).replace(".bw", "")

        feature_map = {}
        start_time = time.time()  # Record the start time
        row_count = 0  # Initialize a counter

        print(f"\nProcessing feature: {basename}")  # Indicate which feature is being processed
        
        for idx, row in adata.obs.iterrows():
            chrom = row['chrom']
            start = row['bin_start']
            end = row['bin_end']
            try:
                value = bw.stats(f"chr{chrom}", start, end, type='mean')[0]
                feature_map[idx] = value
            except:
                print(f"ERROR: {idx}")
                feature_map[idx] = 0.0

            # Print periodic status updates
            row_count += 1
            if row_count % 10000 == 0:  # Adjust the interval as needed
                elapsed_time = time.time() - start_time
                print(f"  Processed {row_count} rows in {elapsed_time:.2f} seconds")

        adata.obs[basename] = adata.obs.index.map(feature_map)
        print(f"  Finished processing {basename}\n")  # Indicate when feature is done

    adata.write(output_path) 
    print("-" * 30)
    print("Analysis Complete")
    print("-" * 30)