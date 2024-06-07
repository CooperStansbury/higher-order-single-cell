import os
import sys
import pandas as pd
import numpy as np

source_path = os.path.abspath("source/")
sys.path.append(source_path)
import utils as ut
import matrix as matrix


def join_fend_info(ref, cell_matrix):
    """Joins fend information onto 'fend1' and 'fend2' columns of the DataFrames.

    Args:
        ref: The DataFrame containing fend, chr, coord, bin information.
        cell_matrix: The DataFrame containing fend1, fend2, and count columns.

    Returns:
        pandas.DataFrame: A new DataFrame with the joined fend information.
    """

    # Merge fend info onto cell_matrix (fend1)
    cell_matrix = cell_matrix.merge(ref, left_on='fend1', right_on='fend', how='left')
    cell_matrix = cell_matrix.rename(columns={'chr': 'fend1_chr', 'coord': 'fend1_coord', 'bin': 'fend1_bin'})

    # Merge fend info onto cell_matrix (fend2)
    cell_matrix = cell_matrix.merge(ref, left_on='fend2', right_on='fend', how='left') 
    cell_matrix = cell_matrix.rename(columns={'chr': 'fend2_chr', 'coord': 'fend2_coord', 'bin': 'fend2_bin'})

    # Keep only necessary columns
    result_df = cell_matrix[['fend1', 'fend2', 'count', 
                             'fend1_chr', 'fend1_coord', 'fend1_bin',
                             'fend2_chr', 'fend2_coord', 'fend2_bin']]

    return result_df


def build_contact_map(df):
    """Processes the Nagano dataset for analysis.

    Args:
        df: The input DataFrame

    Returns:
        pandas.DataFrame: The processed DataFrame.
    """
    # get the bins
    fend1 = df['fend1_bin'].astype(int).unique()
    fend2 = df['fend2_bin'].astype(int).unique()
    intersection = np.intersect1d(fend1, fend2, assume_unique=True)

    # Create a pivot table
    df = pd.pivot_table(
        df, 
        index='fend1_bin', 
        columns='fend2_bin',
        values='count',
        aggfunc='sum',
        fill_value=0
    )
    
    # Filter for bin pairs in both fends
    df = df[intersection]
    df = df[df.index.isin(intersection)]

    # Symmetrize the pivot table
    df = matrix.symmetrize(df)
    return df


if __name__ == "__main__":
    matrix_path = sys.argv[1]  
    fend_path = sys.argv[2]  
    resolution = int(sys.argv[3])
    chrom = sys.argv[4]  
    outpath = sys.argv[5]
    
    # read reference
    chrom_num = chrom.replace("chr", "")
    fend = pd.read_csv(fend_path, sep='\t', low_memory=False)
    fend = fend[fend['chr'] == chrom_num]
    
    # add bin ids
    fend['bin'] = fend['coord'].apply(lambda x: ut.bin_loci(x, resolution)).astype(int)
    
    # read cell data
    cell_matrix = pd.read_csv(matrix_path)
    cell_matrix = join_fend_info(fend, cell_matrix)
    mask = (cell_matrix['fend2_chr'] == chrom_num) & (cell_matrix['fend1_chr'] == chrom_num)
    cell_matrix = cell_matrix[mask]
    
    cell_matrix = build_contact_map(cell_matrix)
    cell_matrix.columns = cell_matrix.columns.astype(str)
    cell_matrix.to_parquet(outpath, index=True)
    

    