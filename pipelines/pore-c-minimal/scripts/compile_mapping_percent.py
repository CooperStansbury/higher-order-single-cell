import pandas as pd
import sys
import os
import re

def extract_mapping_percentages(file_list):
    """
    Extracts mapping percentages from a list of files and returns a DataFrame.

    Args:
        file_list (list): A list of file paths.

    Returns:
        pandas.DataFrame: A DataFrame with columns 'basename' and 'mapping_percentage'.
    """
    
    results = []

    for fpath in file_list:
        with open(fpath, 'r') as file:
            # Get the file name without path
            basename = os.path.basename(fpath) 

            # Read the first line, strip whitespace
            first_line = file.readline().strip()

            # Extract percentage using regex (optimized)
            match = re.search(r"(\d+\.\d+)%", first_line)
            mapping_percentage = float(match.group(1)) if match else None

            results.append({
                'basename': basename,
                'mapping_percentage': mapping_percentage
            })

    return pd.DataFrame(results)  # Create and return the DataFrame

if __name__ == "__main__":
    
    output_path = sys.argv[1] 
    file_list = sys.argv[2:]  
    
    df = extract_mapping_percentages(file_list)
    df.to_csv(output_path, index=False)
   
