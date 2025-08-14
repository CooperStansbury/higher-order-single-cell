import sys
import pandas as pd
import pysam


if __name__ == "__main__":
    bed_path = sys.argv[1]  
    outpath = sys.argv[2]
    
    # prepare the digested reference
    df = pd.read_csv(bed_path, 
                     sep='\t',
                     header=None,
                     low_memory=False,
                     names=['Chromosome', 'Start', 'End'])
    
    df['fragment_id'] = list(range(len(df)))
    df['fragment_length'] = df['End'] - df['Start']
    df.to_parquet(outpath, index=False)