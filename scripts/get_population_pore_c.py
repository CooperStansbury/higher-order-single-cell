import os
import numpy as np
import sys
import pandas as pd
import cooler

source_path = os.path.abspath("source/")
sys.path.append(source_path)
import utils as ut
import matrix as matrix







if __name__ == "__main__":
    resolution = int(sys.argv[1])
    chrom = sys.argv[2]  
    outpath = sys.argv[3] 
    in_paths = sys.argv[4:]
    
    chrom_num = chrom.replace("chr", "")
    
    df = []
    for fpath in in_paths:
        tmp = pd.read_parquet(fpath)
        tmp = ut.filter_and_prepare_data(tmp, resolution, mapq=1)
        tmp = tmp[tmp['chrom'] == chrom_num]
        df.append(tmp)
        
    df = pd.concat(df)
    
    incidence_matrix, _ = ut.process_chromosome_data(df, 
                                                  order_threshold=1, 
                                                  sample_size=None)
    
    incidence_matrix.to_csv(outpath, index=True)
    
    
 

            

            
            

            
            
    
    
    
    
    
    
    
    
    
    
    