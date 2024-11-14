import os
import numpy as np
import sys
import pandas as pd



if __name__ == "__main__":
    chrom_path = sys.argv[1]  
    outpath = sys.argv[2] 
    file_list = sys.argv[3:]
    
    chrom_df = pd.read_csv(chrom_path)
    chroms = chrom_df['chrom'].to_list()
    chrom_starts = dict(zip(chrom_df['chrom'].values, chrom_df['bp_start'].values))
    
    read_columns = [
        'read_name',
        'chrom', 
        'ref_start', 
        'ref_end',
        'is_mapped',
    ]
     
    result = []
    for fpath in file_list:
        basename = os.path.basename(fpath).split(".")[0]
        df = pd.read_parquet(fpath, columns=read_columns)

        # Filtering & Transformations
        df = (
            df[df['is_mapped']]
            .loc[df['chrom'].isin(chroms)]
            .assign(
                basename        = basename,
                local_position  = lambda df: (((df['ref_end'] - df['ref_start']) // 2) + df['ref_start']).astype(int),
                chrom_start     = lambda df: df['chrom'].map(chrom_starts),
                global_position = lambda df: df['chrom_start'].astype(int) + df['local_position'].astype(int),
            )
            .dropna(subset=['global_position'])
            .drop_duplicates()
            .drop(columns=['is_mapped', 'chrom_start'])
        )

        df['ref_start'] = df['ref_start'].astype(int)
        df['ref_end'] = df['ref_end'].astype(int)

        # calculate order and drop singletons efficiently
        df['order'] = df.groupby('read_name')['chrom'].transform('count')
        df = df[df['order'] > 1]
        result.append(df)

    result = pd.concat(result)
    print(f"{result.shape=}")
    
    result.to_parquet(outpath, index=False)


    
 

            

            
            

            
            
    
    
    
    
    
    
    
    
    
    
    