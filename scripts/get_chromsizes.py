import os
import sys
import pandas as pd
import cooler

source_path = os.path.abspath("source/")
sys.path.append(source_path)
import utils as ut
import matrix as matrix




if __name__ == "__main__":
    in_path = sys.argv[1]  
    outpath = sys.argv[2]
    
    resolution = int(1000000) # this doesn't matter  
    
    cool_path = f"{in_path}::resolutions/{resolution}"
    clr = cooler.Cooler(cool_path)
    df = pd.DataFrame(clr.chromsizes).reset_index()
    df.to_csv(outpath, index=False)

            

            
            

            
            
    
    
    
    
    
    
    
    
    
    
    