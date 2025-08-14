import pysam 
import pandas as pd
import os
import sys
import pyranges as pr


if __name__ == "__main__":
    gtf_path = sys.argv[1]  
    outpath = sys.argv[2]

    # load GTF 
    df = pr.read_gtf(gtf_path).df
    
    df = df[df['gene_biotype'] == 'protein_coding']
    df = df[df['Feature'] == 'gene']
    df = df[df['gene_name'].notna()]
    
    columns = [
        'gene_name',
        'gene_id',
        'Chromosome',
        'Start', 
        'End',
    ]
    df = df[columns].drop_duplicates()

    # save to file
    df.to_parquet(outpath, index=False)

   