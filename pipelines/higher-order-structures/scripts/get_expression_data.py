import pandas as pd
import os 
import sys


if __name__ == "__main__":
    gene_table_path = sys.argv[1]  
    outpath = sys.argv[2]  
    file_list = sys.argv[3:]
    
    # load the gene table
    gdf = pd.read_parquet(gene_table_path)
    
    
    # load the expression files
    usecols = [
        'gene_id',
        'TPM', 
    ]

    df = []
    for fpath in file_list:
        file_id = os.path.basename(fpath).replace(".tsv", "")
        # file_id = fpath.
        edf = pd.read_csv(fpath, sep='\t', usecols=usecols)
        edf['gene_id'] = edf['gene_id'].apply(lambda x: x.split(".")[0])

        # filter genes not in GTF
        edf = edf[edf['gene_id'].isin(gdf['gene_id'].unique())]

        # sum expression over all isoforms
        edf = edf.groupby('gene_id').agg(
            TPM = ('TPM', 'sum'),
        ).reset_index()

        edf = edf.set_index('gene_id')
        edf.columns = [f"{file_id}_TPM"]
        df.append(edf)


    df = pd.concat(df, axis=1)
    
    df = pd.merge(df, gdf,
              how='left',
              left_index=True,
              right_on='gene_id')
    
    df.to_parquet(outpath, index=False)
    
    
    