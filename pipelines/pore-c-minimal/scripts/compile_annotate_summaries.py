import pandas as pd
import sys
import os
import glob
import json


def load_json_to_dataframes(file_paths):
    """
    Loads multiple JSON files from a list of file paths, creating three DataFrames:
    - df_cardinality: Cardinality counts with file basenames as columns.
    - df_pair_count: Pair counts with file basenames as columns.
    - df_cis_trans: Cis/Trans counts with file basenames as columns.

    Args:
        file_paths (list): A list of file paths to JSON files.

    Returns:
        tuple: A tuple containing three Pandas DataFrames:
            - df_cardinality: Data on cardinality counts.
            - df_pair_count: Data on pair counts.
            - df_cis_trans: Data on cis/trans counts.
    """

    data = {}
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            basename = os.path.basename(file_path)
            data[basename] = json.load(file)

    # Create DataFrames for each type of data
    df_cardinality = pd.DataFrame({
        basename: pd.Series(d.get('cardinality', {})) for basename, d in data.items()
    })
    df_pair_count = pd.DataFrame({
        basename: pd.Series(d.get('pair_count', {})) for basename, d in data.items()
    })
    df_cis_trans = pd.DataFrame({
        basename: pd.Series(d.get('cis_trans', {})) for basename, d in data.items()
    })
    
    df_cardinality = df_cardinality.reset_index(names='order')
    df_pair_count = df_pair_count.reset_index(names='pair_type')
    df_cis_trans = df_cis_trans.reset_index(names='contact_type')

    return df_cardinality, df_pair_count, df_cis_trans


if __name__ == "__main__":
    
    output_prefix = sys.argv[1] 
    file_list = sys.argv[2:]  
    
    df_cardinality, df_pair_count, df_cis_trans = load_json_to_dataframes(file_list)
    
    card_outpath = f"{output_prefix}cardinality.csv"
    pair_outpath = f"{output_prefix}pair_types.csv"
    ctra_outpath = f"{output_prefix}cis_trans.csv"
    
    df_cardinality.to_csv(card_outpath, index=False)
    df_pair_count.to_csv(pair_outpath, index=False)
    df_cis_trans.to_csv(ctra_outpath, index=False)
   
