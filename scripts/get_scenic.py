import pyranges as pr
import pandas as pd
import sys


if __name__ == "__main__":
    scenic_path = sys.argv[1]  
    outpath = sys.argv[2]
    
    df = pd.read_csv(scenic_path)
    df = df.rename(columns={'Unnamed: 0' : 'gene_name'})
    df = df.set_index('gene_name')
    df.to_parquet(outpath, index=True)