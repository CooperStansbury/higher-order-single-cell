import pandas as pd
import sys
import os



if __name__ == "__main__":
    output = sys.argv[1] 
    file_list = sys.argv[2:]  
    
    res = []

    for file_path in file_list:
        df = pd.read_csv(file_path)

        basename = os.path.basename(file_path)
        row = dict(zip(df['filter'].values, df['rows_filtered_out'].values))
        row['basename'] = basename

        res.append(row)

    res = pd.DataFrame(res)
    res.to_csv(output, index=False)
    