import sys
import os
import pandas as pd
import numpy as np
import glob


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
        tuple: A tuple containing:
            - Pandas DataFrame: with columns ['chrom', 'size', 'bp_start']
            - dict: mapping chromosome names to their start positions (in base pairs)
    """
    chroms = pd.read_csv(fpath, sep='\t', header=None, names=['chrom', 'size'])
    chroms = chroms.head(20) # drop unplaced contigs
    chroms['bp_start'] = chroms['size'].cumsum()
    chroms['bp_start'] = chroms['bp_start'].shift(1).fillna(0).astype(int)
    chrom_starts = dict(zip(chroms['chrom'].values, chroms['bp_start'].values))
    return chroms, chrom_starts



def load_population_pore_c(dpath, chrom_starts, resolution=1e6, chroms=None):
    """
    Loads and processes population Pore-C data from multiple Parquet files.

    This function reads Pore-C alignment data from Parquet files in a specified directory.
    It filters, transforms, and aggregates the data to a desired resolution, keeping only reads
    mapped to specified chromosomes and with a minimum alignment order.

    Args:
        dpath (str): Path to the directory containing the Parquet files.
        chrom_starts (dict): Dictionary mapping chromosome names to their start positions.
        resolution (float, optional): Binning resolution in base pairs. Defaults to 1e6.
        chroms (list, optional): List of chromosome names to include. If None, uses all chromosomes from chrom_starts.

    Returns:
        pandas.DataFrame: Processed Pore-C data with the following columns:
            - read_name
            - align_id
            - order
            - chrom
            - local_position
            - global_bin
            - local_bin
            - basename
    """
    
    file_list = glob.glob(f"{dpath}*")
    
    read_columns = [
        'read_name',
        'align_id',
        'chrom', 
        'ref_start', 
        'ref_end',
        'is_mapped',
    ]
    
    keep_columns = [
        'read_name', 
        'align_id', 
        'order',
        'chrom', 
        'local_position', 
        'global_bin',
        'local_bin', 
        'basename',
    ]
    
    if chroms is None:
        chroms = list(chrom_starts.keys())
    
    df = []    
    for fpath in file_list:
        basename = os.path.basename(fpath).split(".")[0]
        tmp = pd.read_parquet(fpath, columns=read_columns)

        # Filtering & Transformations
        tmp = (
            tmp[tmp['is_mapped']]
            .loc[tmp['chrom'].isin(chroms)]
            .assign(
                local_position  = lambda df: ((df['ref_end'] - df['ref_start']) // 2) + df['ref_start'],
                chrom_start     = lambda df: df['chrom'].map(chrom_starts),
                global_position = lambda df: df['chrom_start'] + df['local_position'],
                global_bin      = lambda df: df['global_position'].apply(lambda x: int(np.ceil(x / resolution))),
                local_bin       = lambda df: df['local_position'].apply(lambda x: int(np.ceil(x / resolution))),
                basename        = basename
            )
            .dropna(subset=['global_bin'])
            .drop_duplicates(subset=['read_name', 'global_bin'])
        )
        
        # calculate order and drop singletons efficiently
        tmp['order'] = tmp.groupby('read_name')['global_bin'].transform('nunique')
        tmp = tmp[tmp['order'] > 1]
        
        tmp = tmp[keep_columns]
        print(basename, tmp.shape)
        df.append(tmp)
        
    df = pd.concat(df)
    return df

