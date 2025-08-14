import os
import sys
import pandas as pd


def load_chrom_sizes(fpath):
    """
    Loads chromosome size information from a tab-separated file.

    This function reads a file containing chromosome names and sizes,
    calculates the cumulative start position for each chromosome, and
    returns the data in two formats:

    1. Pandas DataFrame: containing chromosome names, sizes, and start positions
    2. Dictionary: mapping chromosome names to their start positions

    Args:
        fpath (str): Path to the tab-separated file containing chromosome information.

    Returns:
        - Pandas DataFrame: with columns ['chrom', 'size', 'bp_start']
    """
    chroms = pd.read_csv(fpath, sep='\t', header=None, names=['chrom', 'size'])
    chroms['bp_start'] = chroms['size'].cumsum()
    chroms['bp_start'] = chroms['bp_start'].shift(1).fillna(0).astype(int)
    return chroms



if __name__ == "__main__":
    in_path = sys.argv[1]
    chroms = sys.argv[2]
    outpath = sys.argv[3]
    
    chroms = chroms.split(",")
    
    df = load_chrom_sizes(in_path)
    df = df[df['chrom'].isin(chroms)]
    df.to_csv(outpath, index=False)
    

            

            
            

            
            
    
    
    
    
    
    
    
    
    
    
    