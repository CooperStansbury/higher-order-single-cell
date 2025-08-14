import pandas as pd
import sys
import os



if __name__ == "__main__":
    output = sys.argv[1] 
    file_list = sys.argv[2:]  
    
    res = []
    for file_path in file_list:
        basename = os.path.basename(file_path)

        df = pd.read_parquet(file_path)

        total_reads = len(df)
        unique_reads = df['unique'].sum()

        row = {
            'basename' : basename,
            'total_reads' : total_reads,
            'unique_reads' : unique_reads,
            'duplication_rate' : 1 - (unique_reads / total_reads)
         }

        res.append(row)


    res = pd.DataFrame(res)
    res = res.to_csv(output, index=False)
    