import pandas as pd
import numpy as np
import sys
import pysam
from scipy.sparse import csc_matrix
import pyranges as pr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from pore_c_py import annotate as ann


def get_alignment_df(bamfile):
    """
    Generate a DataFrame from a BAM file assumed to be the output of pore-c-py and MarkDuplicates.

    Parameters:
        bamfile (str): The path to the BAM file.

    Returns:
        pandas.DataFrame: DataFrame containing alignment information.
    """
    # Initialize an empty list to store records
    df_records = []

    # Iterate over concatenated alignments from annotate_alignments
    for concat_walk in ann.annotate_alignments(bamfile):
        # Extract alignment information
        for align in concat_walk:
            read_name, read_start, read_end = align.query_name.split(":")

            # Create a record for the DataFrame
            record = {
                'read_name': read_name,
                'read_start': int(read_start),
                'read_end': int(read_end),
                'length_on_read': int(read_end) - int(read_start),
                'Chromosome': align.reference_name,  # Named for Pyranges join
                'Start': align.reference_start,  # Named for Pyranges join
                'End': align.reference_end,  # Named for Pyranges join
                'mapping_quality': align.mapping_quality,
                'monomer_duplicate': align.is_duplicate,
                'is_mapped': align.is_mapped,
            }
            # Append the record to the list
            df_records.append(record)

    # Create a DataFrame from the list of records
    df = pd.DataFrame(df_records)

    # add a globel monomer-level id 
    df['align_id'] = list(range(len(df)))
    return df


def merge_restriction_fragments(df, fragment_db):
    """
    Merge restriction fragment identifiers with alignments.

    Parameters:
        df (pandas.DataFrame): DataFrame containing alignment information.
        fragment_db (pyranges.PyRanges): PyRanges object containing restriction fragment data.

    Returns:
        pandas.DataFrame: DataFrame containing merged data.
    """
    # Separate records that may be joined
    mask = df['Chromosome'].notna()
    mapped = pr.PyRanges(df[mask].copy())  # Convert to PyRanges
    unmapped = df[~mask].copy()  # Leave as DataFrame

    # Join on overlapping reference coordinates
    overlap = mapped.join(fragment_db,
                          how='left',
                          report_overlap=True,
                          suffix='_fragment')

    # Drop duplicate fragment mappings - keep the longest
    sort_by = ['read_name', 'read_start', 'Overlap']
    sort_order = [True, True, False]
    overlap = overlap.df.sort_values(by=sort_by, ascending=sort_order)
    overlap = overlap.drop_duplicates(subset=['read_name', 'read_start'])

    # Join all records back together and resort
    df = pd.concat([overlap, unmapped])
    df = df.sort_values(by=['read_name', 'read_start'])
    return df


def group_by_proximity(data, threshold):

    """Groups elements in a list within a certain threshold of each other.

    Args:
    data: A list of integers.
    threshold: The maximum difference between two elements to be considered in the same group.

    Returns:
    A list of group labels (integers), where each element represents the group an element in the original data belongs to.
    """
    group_id = 0
    group_labels = [None] * len(data)
    for i, num in enumerate(data):
        if not group_labels[i]:
            group_id += 1
            group_labels[i] = group_id
            for j in range(i + 1, len(data)):
                if abs(num - data[j]) <= threshold and not group_labels[j]:
                    group_labels[j] = group_id
    return group_labels


def flag_monomer_proximity(df, id_column, select_by, threshold):
    """
    Identifies monomers to keep based on proximity grouping within a DataFrame.

    Args:
      df (pandas.DataFrame): The input DataFrame.
      id_column (str): The column containing unique identifiers for rows.
      select_by (str): The column containing the selection criteria for monomers.
      threshold (int): The maximum difference for elements to be considered close.

    Returns:
      dict: A dictionary mapping unique identifiers to boolean flags indicating if 
          the row should be retained.

    Note: this function overwrites variables in memory in a shallow attempt to 
    reduce memory footprint
    """
    monomer_flags = {}
    for (read_name, chrom), group in df.groupby(['read_name', 'chrom'], sort=False):
        group = group.sort_values(by=id_column)
        p_group = group_by_proximity(group[id_column].tolist(), threshold)

        # Combine data with group labels
        group = pd.DataFrame({
            'align_id' : group['align_id'],
            id_column: group[id_column],
            select_by: group[select_by],
            'p_group': p_group
        })

        # Identify the first occurrence within each proximity group to retain
        group['retain'] = group.groupby('p_group')[select_by].transform(pd.Series.rank, method='first', ascending=False) == 1
        monomer_map = dict(zip(group['align_id'], group['retain']))
        monomer_flags.update(monomer_map)
    return monomer_flags


def concatamer2list(group):
    """
    Efficiently joins fragment ID codes into a semicolon-separated string, excluding -1.
    
    Parameters:
        group (pandas.Series): Series containing fragment ID codes.
    
    Returns:
        str: Semicolon-separated string of fragment ID codes.
    """
    return ";".join(sorted(set(str(x) for x in group.dropna())))
    


def flag_read_duplicates(df, id_column='fragment_id', 
                         select_by='mapping_quality', 
                         drop_low=1):
    """
    Assign read-level duplication flags optimized for efficiency and readability.

    Parameters:
        df (pandas.DataFrame): DataFrame containing alignment information.
        id_column (str): Column name for the fragment ID. Default is 'fragment_id'.
        select_by (str): Column name for selecting reads. Default is 'mapping_quality'.
        drop_low (int): Minimum number of occurrences to consider. Default is 1.

    Returns:
        tuple: Tuple containing uniqueness map and read group map.

    Notes:
        - This function assumes that the DataFrame contains a boolean column 'is_mapped'.

    """
    # Filter and copy the DataFrame for safety
    df = df[df['is_mapped']].copy()

    # Assign categorical codes to fragment IDs
    df['code'] = df[id_column].astype('category').cat.codes

    # Aggregate information efficiently
    df = (
        df.groupby('read_name')
        .agg(sorter=(select_by, 'mean'), 
             codelist=('code', concatamer2list), 
             n_code=(id_column, 'nunique'))
        .reset_index(drop=False)
    )

    # Filter and assign uniqueness flags in a vectorized manner
    df = df[df['n_code'] > drop_low]
    df['read_unique'] = df.groupby('codelist')['sorter'].transform(pd.Series.rank, method='first', ascending=False) == 1
    df['read_group'] = df.groupby('codelist').ngroup() # add read_group id

    # Return dictionaries for efficient access
    uniqueness_map = dict(zip(df['read_name'].values, df['read_unique'].values))
    read_group_map = dict(zip(df['read_name'].values, df['read_group'].values))
    return uniqueness_map, read_group_map


def flag_near_duplicates(df, id_column='fragment_id', threshold=0.6):
    """
    Group near duplicates in a DataFrame based on feature similarity.

    Jaccard similarity coefficient, defined as the size of the
    intersection divided by the size of the union of two label sets

    Parameters:
        df (pd.DataFrame): DataFrame with rows representing samples and columns representing binary features.
        threshold (float): Threshold for similarity (distance) to consider rows as near duplicates. Defaults to 0.5.
            lower numbers are more stringent, high numbers are more liberal when flagging reads as near duplicates

    Returns:
        tuple: Tuple containing uniqueness map and read group map.
    """
    # Filter based on combined conditions
    mask = (df['exact_unique'] & 
          (df.groupby('read_name')[id_column].transform('nunique') > 2) & 
          (df.groupby(id_column)['read_name'].transform('nunique') > 2))
    df = df[mask].reset_index(drop=True)


    # pivot the dataframe so that each row
    # represent a single read and each column represents
    # each mapped monomer
    df['val'] = 1
    df = pd.pivot_table(df, 
                        index='read_name',
                        columns=id_column,
                        values='val',
                        fill_value=0).astype(bool)
     
    # sort by "cardinality" so that we can bias
    # selection of higher-order, i.e., more 'complete' reads 
    df = df.sort_index(level=id_column, 
                       ascending=False)

    # store read names
    read_names = df.index
    
    # Convert df to sparse CSR matrix
    df = csc_matrix(df)

    # Calculate pairwise Jaccard distance between rows using sparse operations
    distances = pdist(df.toarray(), metric='jaccard')

    # if no eligible reads, return
    if distances.size == 0:
        return {}, {}

    # Perform hierarchical clustering to identify 
    # near duplicates
    clusters = fcluster(linkage(distances, 
                                method='complete', 
                                metric='jaccard'), 
                        t=threshold, 
                        criterion='distance')

   # Create final DataFrame efficiently
    dupes = pd.DataFrame({
        'read_name': read_names,
        'read_group': clusters,
        'order': read_names.map(read_names.value_counts()),  # Get 'order' efficiently
    })

    # flag the largest concatemer as the one kept
    # TODO: this criteria biases concatemer selection to potentially over-digested
    # sets of monomers - potential fix to weight selection of
    # 'best' concatemer of duplicate group by mappability and monomer
    # total bases covered on read/reference
    dupes['exact_unique'] = dupes.groupby('read_group')['order'].transform(pd.Series.rank, method='first', ascending=False) == 1

    uniqueness_map = dict(zip(dupes['read_name'].values, dupes['exact_unique'].values))
    read_group_map = dict(zip(dupes['read_name'].values, dupes['read_group'].values))
    return uniqueness_map, read_group_map

