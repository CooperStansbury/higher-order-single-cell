import pyranges as pr
import pandas as pd
import sys


if __name__ == "__main__":
    gtf_path = sys.argv[1]  
    scenic_path = sys.argv[2]  
    outpath = sys.argv[3]
    
    # load scenic column names
    tfs = pd.read_parquet(scenic_path).columns.tolist()
        
    keep_columns = [
        'gene_id', 
        'gene_name', 
        'gene_source',
        'gene_biotype', 
        'Chromosome',
        'Start',
        'End',
    ]

    fpath = "/scratch/indikar_root/indikar1/shared_data/scpore_c/gtf/GRCm39.gtf.gz"
    df = pr.read_gtf(gtf_path).as_df()

    df = df[df['Feature'] == 'gene']
    df = df[keep_columns]
    df = df[df['gene_name'].notna()]
    df = df.drop_duplicates()
    
    # add some informative columns
    df['length'] = df['End'] - df['Start']
    df['midpoint'] = ((df['End'] - df['Start']) / 2) + df['Start']
    df['midpoint'] = df['midpoint'].astype(int)
    df['is_tf'] = df['gene_name'].str.upper().isin(tfs)
    
    df.to_parquet(outpath, index=False)