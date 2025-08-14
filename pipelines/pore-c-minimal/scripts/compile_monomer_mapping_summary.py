import pandas as pd
import sys
import os
import numpy as np

def get_mapped_read_counts(df):
    """Calculates the count of reads bucketed by the number of mapped monomers.

    Args:
        df (pandas.DataFrame): The input DataFrame containing 'read_name' and 'is_mapped' columns.

    Returns:
        dict: A dictionary where keys are like 'reads_with_X_mapped' (X = 0, 1, 2, 3+)
              and values are the corresponding counts.
    """

    # Group by read name, sum is_mapped
    monomers_mapped = df.groupby('read_name')['is_mapped'].sum()

    # Create bins and labels (same as your code)
    bins = [-np.inf, 0, 1, 2, np.inf]
    labels = ['0', '1', '2', '3+']

    # Bucket the values, count occurrences, and reset index
    monomers_mapped = pd.cut(monomers_mapped, bins=bins, labels=labels, right=False)
    monomers_mapped = monomers_mapped.value_counts().reset_index()

    # Format the 'index' column to match your desired output
    monomers_mapped['is_mapped'] = "reads_with_" + monomers_mapped['is_mapped'].astype(str) + "_mapped"

    # Convert to dictionary (same as your code)
    return dict(zip(monomers_mapped['is_mapped'].values, monomers_mapped['count'].values))


def get_summary_row(df):
    """A function to get key summary mertics from an alignment table """
    results = {}
    results['total_reads'] = df['read_name'].nunique()
    monomers_mapped = get_mapped_read_counts(df)
    
    results = results | monomers_mapped
    
    return results



if __name__ == "__main__":
    
    output_path = sys.argv[1] 
    file_list = sys.argv[2:]  
    
    res = []

    for file_path in file_list:
        basename = os.path.basename(file_path)

        df = pd.read_parquet(file_path)
        row = get_summary_row(df)
        row['basename'] = basename
        res.append(row)

    res = pd.DataFrame(res)
    
    res.to_csv(output_path, index=False)
    

    
   
   
