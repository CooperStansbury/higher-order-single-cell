import pandas as pd
import sys
import os
import numpy as np
import pairtools
import pairtools.lib.headerops as phead


def load_pairs(fpath, comment_char="#"):
    """Loads a pairs table from a tab-separated file.

    Args:
        fpath (str): The file path to the pairs table.
        comment_char (str, optional): The character indicating comment lines. 
                                     Defaults to "#".

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the pairs table data.
    """
    header = phead.get_header(open(fpath))[0][-1]
    header = header.replace("#columns: ", "")
    header = header.split(" ")
    
    df = pd.read_csv(fpath, 
                     sep='\t', 
                     header=None, 
                     names=header, 
                     comment="#")
    return df


def get_basic_stats(df):
    """A function to get basic statistics """
    stats = {}
    stats['n_contacts'] = len(df)
    stats['n_cis'] = (df['chrom1'] == df['chrom2']).sum()
    stats['n_trans'] = (df['chrom1'] != df['chrom2']).sum()
    return stats


def get_read_stats(df):
    """A function to get read statistics """
    stats = {}
    stats['unique_reads'] = df['readID'].nunique()
    
    orders = df['readID'].value_counts()
    orders = np.where(orders == 1, 'pairwise', 'higher_order')
    orders = pd.Series(orders).value_counts()
    orders = dict(zip(orders.index, orders.values))
    
    stats = stats | orders
    
    return stats



if __name__ == "__main__":
    
    basic_output = sys.argv[1] 
    read_output =  sys.argv[2] 
    file_list = sys.argv[3:]  
    
    basic_stats = []
    read_stats = []

    for file_path in file_list:
        basename = os.path.basename(file_path)
        df = load_pairs(file_path)

        row = {
            'basename' : basename
        }

        # basic statistics
        basic_row = row | get_basic_stats(df)
        basic_stats.append(basic_row)

        # read-level statistics
        read_row = row | get_read_stats(df)
        read_stats.append(read_row)

    # compile dataframes 
    basic_stats = pd.DataFrame(basic_stats)   
    read_stats = pd.DataFrame(read_stats)   
    
    # save dataframes 
    basic_stats.to_csv(basic_output, index=False)
    read_stats.to_csv(read_output, index=False)
   
   
