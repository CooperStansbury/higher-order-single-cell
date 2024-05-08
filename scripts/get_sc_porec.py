import os
import sys
import pandas as pd
import numpy as np

source_path = os.path.abspath("source/")
sys.path.append(source_path)
import utils as ut
import matrix as matrix



if __name__ == "__main__":
    table_path = sys.argv[1]  
    resolution = int(sys.argv[2])
    chrom = sys.argv[3]  
    outpath = sys.argv[4]
    
    # load and filter PCR duplicates
    df = pd.read_parquet(table_path)
    df = df[df['near_unique'] & df['exact_unique']]
    
    df = ut.filter_and_prepare_porec_data(df, resolution, mapq=60)
    incidence_matrix, _ = ut.process_chromosome_data(df, 
                                              order_threshold=1, 
                                              sample_size=None)
    
    incidence_matrix.to_csv(outpath, index=True)
    
    
    
    

    