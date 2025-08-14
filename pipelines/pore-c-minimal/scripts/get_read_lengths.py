import queue
import sys
import pandas as pd
import pysam


if __name__ == "__main__":
    fastq_path = sys.argv[1]  
    outpath = sys.argv[2]
    
    records = []
    with pysam.FastxFile(fastq_path, "r") as f:
        for read in f:
            record = {
                'read_name' : read.name,
                'read_length' : len(read.sequence),
            }

            records.append(record)
          
    df = pd.DataFrame(records)
    df.to_parquet(outpath, index=False)
    


    