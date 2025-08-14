import pyranges as pr
import pandas as pd
import os
import glob
import sys
import numpy as np

source_path = os.path.abspath("source/")
sys.path.append(source_path)
import utils as ut



if __name__ == "__main__":
    gene_table_path = sys.argv[1]
    chrom = sys.argv[2]  
    outpath = sys.argv[3] 
    file_list = sys.argv[4:]
    
    chrom_num = chrom.replace("chr", "")
    
    """ Load the gene table """
    columns = [
        'gene_name',
        'gene_biotype',
        'is_tf',
        'Chromosome',
        'Start',
        'End',

    ]
    gdf = pd.read_parquet(gene_table_path, columns=columns)
    gdf = gdf[gdf['Chromosome'] == chrom_num]
    gdf = pr.PyRanges(gdf)
    
    
    """ Load the pore-c data """
    slack = 1000
    columns = [
        'read_name',
        'chrom',
        'ref_start',
        'ref_end',
    ]

    res = []

    for fpath in file_list:
        file_id = os.path.basename(fpath)
        tmp = pd.read_parquet(fpath, columns=columns)
        tmp.columns = ['read_name', 'Chromosome', 'Start', 'End']

        # drop missing mappings
        tmp = tmp[tmp['Chromosome'] == chrom_num]
        tmp = pr.PyRanges(tmp)

        tmp = tmp.join(gdf, 
                       slack=slack,
                       suffix='_gene',
                       report_overlap=True)

        tmp = tmp.df

        tmp['gene_order'] = tmp.groupby('read_name')['gene_name'].transform('nunique')
        tmp = tmp[tmp['gene_order'] > 1]
        tmp['file'] = file_id
        res.append(tmp)


    res = pd.concat(res)
    res = res.sort_values(by='read_name')
    res['read_code'] = res['read_name'].astype('category').cat.codes
    
    
    """Make the incidence matrix """
    val = np.ones(len(res))
    incidence_matrix = ut.incidence_by_pivot(res, 'read_code', 'gene_name', val)
    
    incidence_matrix.columns = incidence_matrix.columns.astype(str)
    incidence_matrix.to_parquet(outpath, index=True)
    
    
 

            

            
            

            
            
    
    
    
    
    
    
    
    
    
    
    