import pandas as pd
import os 
import sys
import numpy as np
from datasketch import MinHash, MinHashLSH


def preliminary_filters(df):
    """
    Filters a DataFrame to retain only rows with mapped monomers.

    This function applies a filtering step to remove entries corresponding
    to unmapped monomers within the provided DataFrame. The filtering is 
    based on the boolean column 'is_mapped'.

    Args:
        df (pd.DataFrame): The input DataFrame containing monomer data,
                           including an 'is_mapped' column.

    Returns:
        pd.DataFrame: The filtered DataFrame containing only rows where
                      'is_mapped' is True.
    """
    df = df[df['is_mapped']]
    return df


def concatamer2list(group):
    """
    Efficiently joins fragment ID codes into a semicolon-separated string, excluding -1.
    
    Parameters:
        group (pandas.Series): Series containing fragment ID codes.
    
    Returns:
        str: Semicolon-separated string of fragment ID codes.
    """
    return ";".join(sorted(set(str(x) for x in group.dropna())))


def flag_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Identifies and flags exact duplicate reads within a DataFrame.

    Exact duplicates are defined as reads sharing the same set of fragment IDs.

    Args:
        df: A pandas DataFrame with columns:
            - `read_name`
            - `fragment_id`
            - `mapping_quality`

    Returns:
        A pandas DataFrame with additional columns:
            - `mapq`: Mean mapping quality of the read.
            - `order`: Number of unique fragment IDs in the read.
            - `exact_duplicate_group`: ID of the exact duplicate group.
            - `fragments`: Concatenated list of unique fragment IDs.

    """
    # Assign categorical codes to fragment IDs for efficiency.
    df['code'] = df['fragment_id'].astype('category').cat.codes

    # Aggregate information by read name.
    df = (
        df.groupby('read_name')
        .agg(mapq=('mapping_quality', 'mean'),
             fragments=('code', concatamer2list),
             order=('fragment_id', 'nunique'))
        .reset_index(drop=False)
    )

    # Identify exact duplicate groups using fragment codes.
    df['exact_duplicate_group'] = df.groupby('fragments').ngroup()

    # Select and return relevant columns.
    return df[['read_name', 'mapq', 'order', 'exact_duplicate_group', 'fragments']]


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
    return similar_groups


def annotate_near_duplicates(df: pd.DataFrame, near_read_groups: list) -> pd.DataFrame:
    """
    Annotates near-duplicate reads within a DataFrame.

    This function identifies and assigns group IDs to near-duplicate reads, 
    where near-duplicates are defined by sharing at least one fragment ID 
    with other reads in the same group. It also assigns unique IDs to reads 
    that do not belong to any near-duplicate group.

    Args:
        df (pd.DataFrame): The input DataFrame containing read data,
                           expected to have a 'fragments' column with 
                           concatenated fragment IDs.
        near_read_groups (list): A list of lists, where each inner list 
                                represents a group of near-duplicate reads
                                identified by their 'fragments'.

    Returns:
        pd.DataFrame: The input DataFrame with an additional column 
                      'near_duplicate_group' containing the assigned group
                      IDs. Reads not belonging to any near-duplicate group
                      will have unique IDs.

    Example:
        >>> df = pd.DataFrame({'fragments': [['0', '1'], ['0', '2'], ['3']]})
        >>> near_read_groups = [['0', '1'], ['0', '2']]
        >>> annotate_near_duplicates(df, near_read_groups)
           fragments  near_duplicate_group
        0  [0, 1]                      0
        1  [0, 2]                      1
        2     [3]                      2
    """
    df['near_duplicate_group'] = -1  # Initialize the new column
    for i, group in enumerate(near_read_groups):
        codelist = list(set(group))  # Get unique fragment IDs in the group
        df['near_duplicate_group'] = np.where(
            df['fragments'].isin(group), i, df['near_duplicate_group']
        )  # Assign group ID if read is in the group

    # Assign unique IDs to reads not in any near-duplicate group
    max_id = df['near_duplicate_group'].max()
    unknown_mask = df['near_duplicate_group'] == -1
    new_ids = range(max_id + 1, max_id + 1 + unknown_mask.sum())
    df.loc[unknown_mask, 'near_duplicate_group'] = new_ids

    return df


def annotate_uniques(df: pd.DataFrame) -> pd.DataFrame:
    """Annotates unique reads within near-duplicate groups in a DataFrame.

    This function identifies the read with the highest mapping quality (`mapq`) 
    within each near-duplicate group and marks it as unique (`unique` = True).
    All other reads within the same group are marked as not unique (`unique` = False).

    Args:
        df (pd.DataFrame): The input DataFrame containing read data,
                           expected to have columns:
                           - `near_duplicate_group`
                           - `mapq`

    Returns:
        pd.DataFrame: The input DataFrame with an additional boolean column 
                      `unique` indicating whether a read is the unique 
                      representative of its near-duplicate group.

    Example:
        >>> df = pd.DataFrame({'near_duplicate_group': [0, 0, 1, 1],
        ...                    'mapq': [30, 40, 20, 25]})
        >>> annotate_uniques(df)
           near_duplicate_group  mapq  unique
        0                      0    30  False
        1                      0    40   True
        2                      1    20  False
        3                      1    25   True
    """
    df['unique'] = (df.groupby('near_duplicate_group')['mapq']
                    .transform(pd.Series.rank, method='first', ascending=False) == 1)
    return df

if __name__ == "__main__":
    align_table_path = sys.argv[1]  
    outpath = sys.argv[2]
    
    # load and preliminary filters
    df = pd.read_parquet(align_table_path)
    df = preliminary_filters(df) # drop unmapped monomers
    
    # collapses to a read-level dataframe
    df = flag_exact_duplicates(df)
    
    # get near duplicate groupings
    read_groups = find_similar_entries_minhash(df['fragments'].values, threshold=0.6)
    
    df = annotate_near_duplicates(df, read_groups)
    df = annotate_uniques(df)
    
    # store outputs
    df.to_parquet(outpath, index=False)
    
    
