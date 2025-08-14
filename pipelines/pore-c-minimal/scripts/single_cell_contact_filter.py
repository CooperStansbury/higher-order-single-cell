import pandas as pd
import os 
import sys
import numpy as np
from pyranges import PyRanges
import pairtools
import pairtools.lib.headerops as phead

def load_pairs(fpath, comment_char="#"):
    """Loads a pairs table from a tab-separated file.

    Args:
        fpath (str): The file path to the pairs table.
        comment_char (str, optional): The character indicating comment lines. 
                                     Defaults to "#".

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the pairs table data.
    """
    header = phead.get_header(open(fpath))[0][-1]
    header = header.replace("#columns: ", "")
    header = header.split(" ")
    
    df = pd.read_csv(fpath, 
                     sep='\t', 
                     header=None, 
                     names=header, 
                     comment="#")
    return df



def filter_unmapped(df):
    """A function to filter the unmapped reads """
    df = df[(df['chrom1'] != "!") & (df['chrom2'] != "!")].copy()
    return df


def filter_adjacent(df, tolerance=1):
    """A function to filter out fragments
    adjacent on the reference """
    return df[df['rfrag1'].sub(df['rfrag1'].shift(1)).abs() > tolerance].copy()


def filter_close_contact(df, bp=1000):
    """A function to filter contacts which are close"""
    return df[~((df['chrom1'] == df['chrom2']) & (df['pos1'].sub(df['pos2']).abs() < bp))].copy()


def filter_duplicate_contacts(df):
    return (df.assign(rfrag_diff=df['rfrag1'].sub(df['rfrag2']).abs())
              .sort_values('rfrag_diff')
              .drop_duplicates(subset=['rfrag1', 'rfrag2'], keep='first')
              .drop(columns=['rfrag_diff']))


def filter_promiscuous_fragments(df, threshold = 10):
    """Filters out contacts involving promiscuous fragments from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing contact information with 'rfrag1' and 'rfrag2' columns.
        threshold (int, optional): The minimum number of interactions for a fragment to be considered promiscuous. Defaults to 10.

    Returns:
        pd.DataFrame: The filtered DataFrame with contacts involving promiscuous fragments removed.
    """

    fragment_counts = (pd.concat([df["rfrag1"], df["rfrag2"]])
                         .value_counts()
                         .reset_index(name="freq")
                         .rename(columns={"index": "fragment_id"}))

    promiscuous_fragments = fragment_counts[fragment_counts["freq"] > threshold]["fragment_id"].tolist()

    return df[~(df["rfrag1"].isin(promiscuous_fragments) | df["rfrag2"].isin(promiscuous_fragments))]


def filter_isolated_fragments(df, max_distance = 1_000_000):
    """Filters out contacts involving isolated fragments from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing contact information with 
            'rfrag1', 'rfrag2', 'chrom1', 'pos1', 'chrom2', 'pos2' columns.
        max_distance (int, optional): The maximum distance to another fragment for a fragment to be considered non-isolated. Defaults to 1,000,000.

    Returns:
        pd.DataFrame: The filtered DataFrame with contacts involving isolated fragments removed.
    """
    
    # Create a combined DataFrame with fragment information
    fragment_data = pd.concat([
        df[['rfrag1', 'chrom1', 'rfrag_start1', 'rfrag_end1']].rename(columns={'rfrag1': 'id', 'chrom1': 'Chromosome', 'rfrag_start1': 'Start', 'rfrag_end1': 'End'}),
        df[['rfrag2', 'chrom2', 'rfrag_start2', 'rfrag_end2']].rename(columns={'rfrag2': 'id', 'chrom2': 'Chromosome', 'rfrag_start2': 'Start', 'rfrag_end2': 'End'}),
    ]).drop_duplicates()

    # Create PyRanges object
    frag_ranges = PyRanges(fragment_data)

    # Find nearest neighbors
    nearest_fragments = frag_ranges.nearest(frag_ranges)

    # Identify isolated fragments
    isolated_fragments = nearest_fragments.df[nearest_fragments.df["Distance"] > max_distance]["id"].tolist()

    # Filter out contacts involving isolated fragments
    return df[~(df["rfrag1"].isin(isolated_fragments) | df["rfrag2"].isin(isolated_fragments))]


def contact_filter(df):
    """Filters contacts, returning the filtered DataFrame and a summary of filter results.

    Args:
        df (pandas.DataFrame): DataFrame with contact information.

    Returns:
        tuple: A tuple containing:
            - pandas.DataFrame: The final filtered DataFrame.
            - pandas.DataFrame: A DataFrame summarizing the number of rows removed by each filter.
    """

    df = df.reset_index(drop=True).copy()  # Ensure a copy to avoid SettingWithCopyWarning
    summary_data = []

    filters = [
        (filter_unmapped, "unmapped"),
        (filter_adjacent, "adjacent"),
        (filter_close_contact, "close_contact"),
        (filter_duplicate_contacts, "duplicate"),
        (filter_promiscuous_fragments, "promiscuous"),
        (filter_isolated_fragments, "isolated"),
    ]

    for filter_func, filter_name in filters:
        original_size = len(df)
        df = filter_func(df)  # Apply the filter
        filtered_out_count = original_size - len(df) # Calculate rows removed
        summary_data.append({"filter": filter_name, "rows_filtered_out": filtered_out_count}) # Store summary

    summary_df = pd.DataFrame(summary_data)  # Create summary DataFrame
    return df, summary_df


def write_pairs(df, header, outpath):
    """A function to write a pairs file """
    with open(outpath, "a") as file:  # Open in append mode
        for line in header:
            file.write(line + "\n") 
        
    df.to_csv(outpath, mode='a', header=None, sep="\t", index=False)
    


if __name__ == "__main__":
    
    pairs_path = sys.argv[1]  
    output = sys.argv[2]
    summary_output = sys.argv[3]
    
    # load the data
    df = load_pairs(pairs_path)
    
    # run filters
    df, summary = contact_filter(df)
    
    # save outputs
    # get the header record
    header = phead.get_header(open(pairs_path))[0]
    write_pairs(df, header, output)
    
    # store the summary
    summary.to_csv(summary_output, index=False)
    
    
    
    
    
    
    
