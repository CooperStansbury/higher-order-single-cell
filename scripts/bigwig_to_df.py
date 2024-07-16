import pyranges as pr
import pandas as pd
import sys
import os
import pyBigWig
import numpy as np


if __name__ == "__main__":
    outpath = sys.argv[1]  
    chrom = sys.argv[2]
    resolution = int(sys.argv[3])
    file_list = sys.argv[4:]
    
    
    df = []
    for fpath in file_list:
        file_id = os.path.basename(fpath).replace(".bw", "")

        bw = pyBigWig.open(fpath)  # Replace with your BigWig file path

        # Get chromosome sizes
        chrom_length = bw.chroms()[chrom]
        n_bins = int(np.ceil(chrom_length / resolution))
        
        stats = bw.stats(chrom, nBins=n_bins, type='sum', exact=True)

        bwdf = pd.DataFrame({file_id : stats , },
                          index=list(range(n_bins)),)
        df.append(bwdf)

    df = pd.concat(df, axis=1)
    df = df.reset_index()
    
    df.to_parquet(outpath, index=False)