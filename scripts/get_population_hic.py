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
    resolution = int(sys.argv[2])
    chrom = sys.argv[3]  
    outpath = sys.argv[4]  
    
    cool_path = f"{in_path}::resolutions/{resolution}"
    clr = cooler.Cooler(cool_path)

    A = clr.matrix(balance=False).fetch(str(chrom))[:]
    A = pd.DataFrame(A)
    
    A.to_csv(outpath, index=True)

    print(f"{A.shape}")

            

            
            

            
            
    
    
    
    
    
    
    
    
    
    
    