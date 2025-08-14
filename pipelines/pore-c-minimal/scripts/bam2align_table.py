import pandas as pd
import numpy as np
import sys
import pysam
import pyranges as pr
from pore_c_py import annotate
import align_table_tools as algn


def annotate_restriction(df, fragment_db):
    """A wrapper function to align restriction 
    fragments """
    fragment_db = pd.read_parquet(db_path)
    fragment_db = pr.PyRanges(fragment_db) # convert to pyranges

    df = algn.merge_restriction_fragments(df, fragment_db)

    # clean up columns
    new_names = {
        'Chromosome': 'chrom', 
        'Start': 'ref_start',
        'End': 'ref_end',
        'Start_fragment' : 'fragment_start',
        'End_fragment' : 'fragment_end',
    }
    df = df.rename(columns=new_names)
    df = df.reset_index(drop=True)
    return df

    
def clean_up(df):
    """A function to set the final column inclusion/order """
    column_order = [
        'read_name', 
        'align_id',
        'read_start', 
        'read_end', 
        'length_on_read',
        'chrom', 
        'ref_start',
        'ref_end',
        'fragment_id',
        'fragment_start',
        'fragment_end',
        'fragment_length',
        'monomer_duplicate',
        'is_mapped',
        'mapping_quality', 
    ]

    df = df[column_order]
    df = df.sort_values(by=['read_name', 'read_start'])
    df = df.reset_index(drop=True)
    return df
    


if __name__ == "__main__":
    
    bam_path = sys.argv[1] 
    db_path = sys.argv[2]  
    outpath = sys.argv[3]

    # load alignment data tables
    bamfile = pysam.AlignmentFile(bam_path)
    df = algn.get_alignment_df(bamfile)

    print(f"raw {df.shape=}")
    print(f"raw {df['read_name'].nunique()=}")

    # load restriction fragment database
    fragment_db = pd.read_parquet(db_path)
    fragment_db = pr.PyRanges(fragment_db) # convert to pyranges

    # resolve restriction fragments
    df = annotate_restriction(df, fragment_db)

    # final clean-up and save 
    df = clean_up(df)

    df.to_parquet(outpath, index=False)
    

