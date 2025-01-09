import os
import sys
import pandas as pd
import numpy as np
import glob
import time
from scipy.sparse import csr_matrix
import anndata as an
import scanpy as sc
from datasketch import MinHash, MinHashLSH


def find_similar_entries_minhash(arr, threshold=0.5, num_perm=128):
    """Finds similar entries in an array using MinHash for Jaccard similarity approximation.

    Args:
        arr (numpy.ndarray): The input array of strings representing sets.
        threshold (float): The minimum Jaccard similarity for two sets to be considered similar.
        num_perm (int): The number of permutations used for MinHash.

    Returns:
        list: A list of lists, where each inner list contains similar entries.
    """
    
    minhashes = {}
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    for i, entry in enumerate(arr):
        m = MinHash(num_perm=num_perm)
        for val in map(int, entry.split(';')):
            m.update(str(val).encode('utf8'))  # Correct encoding for integers
        lsh.insert(str(i), m)
        minhashes[i] = m

    similar_groups = []
    processed = set()
    for i in range(len(arr)):
        if i in processed:
            continue
        similar = [arr[i]]
        result = lsh.query(minhashes[i])  
        for key in result:
            j = int(key)
            if j != i and j not in processed:  # Avoid self and duplicates
                similar.append(arr[j])
                processed.add(j)  # Mark as processed
        if len(similar) > 1:
            similar_groups.append(similar)

    # we only need the unique bin pairs to flag duplicates
    similar_groups = [list(set(x)) for x in similar_groups] 
    similar_groups = [item for sublist in similar_groups for item in sublist]
    return similar_groups

def get_edgelist(hyperedge):
    """Converts a hyperedge (represented as a binary array) to an edge list string.
    
    Args:
        hyperedge (numpy.ndarray): A binary array representing the hyperedge, 
                               where 1 indicates the presence of a node.
    
    Returns:
        str: A semicolon-separated string of node indices.
    """
    nodes_in_hyperedge = sorted(np.nonzero(hyperedge)[0])
    return ";".join(map(str, nodes_in_hyperedge))


def annotate_column(df, reference_list):
    """Annotates a pandas Series (DataFrame column) with the index of the element 
    from a reference list if the element appears in the reference list and -1 
    otherwise.
    
    Args:
        df (pandas.Series): The input Series (DataFrame column) containing the elements.
    reference_list (list): The reference list of elements.
    
    Returns:
        pandas.Series: A series containing the annotations.
    """
    def get_index(row):
        try:
          return reference_list.index(row)
        except ValueError:
          return -1
    return df.apply(get_index)



if __name__ == "__main__":
    anndata_path = sys.argv[1]
    outpath = sys.argv[2]

    threshold = 0.5

    start_time = time.time()  # Record the start time
    adata = sc.read_h5ad(anndata_path)
    end_time = time.time()  # Record the end time
    print(f"Time taken to read the file: {end_time - start_time:.2f} seconds")
    sc.logging.print_memory_usage()

    """ Loop through each single-cell and find and mark
    duplicated hyperedges """
    result = []
    for cell_id, group in adata.var.groupby('basename'):
        # data structure for the hyperedges
        scdata =  adata[:, group.index].copy()

        if len(group) == 0:
            continue
            
        df = pd.DataFrame.sparse.from_spmatrix(
            scdata.X,
            index=scdata.obs_names,
            columns=scdata.var_names,
        ).T 
    
        # data structure for the results
        annot = pd.DataFrame({
            'read_id' : df.index,
            'mapping_quality' : scdata.var['mean_mapq'].values
        })
        annot['cell_id'] = cell_id
        annot['exactly_unique'] = np.ravel(~df.duplicated())
    
        # get hyperedges for hasing 
        hyperedges = df.apply(get_edgelist, axis=1)
    
        # find and annotate nearly-identical read groups
        duplicated_hyperedges = find_similar_entries_minhash(
            hyperedges, 
            threshold=threshold,
        )
        annot['approximately_unique'] = np.ravel(~hyperedges.isin(duplicated_hyperedges))
        annot['read_group'] = np.ravel(annotate_column(hyperedges, duplicated_hyperedges))
        
        # mark duplicates
        annot['unique'] = (annot.groupby('read_group')['mapping_quality'].transform(
            pd.Series.rank,
            method='first',
            ascending=False) == 1)
    
        # make sure that truly unique reads are retained
        annot['unique'] = np.where(annot['exactly_unique'], True, annot['unique'])
        annot['unique'] = np.where(annot['approximately_unique'], True, annot['unique'])
        result.append(annot)

    # structure the results and save
    result = pd.concat(result)
    result.to_parquet(outpath, index=False)

    

    
