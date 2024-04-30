import sys
import pandas as pd
import utils as ut


if __name__ == "__main__":
    ref_path = sys.argv[1]  
    res_path = sys.argv[2]  
    out_path = sys.argv[3]
    
    # read reference
    ref = pd.read_csv(ref_path, sep='\t')
    print(ref.head())
    
    # read resolutions
    res = ut.read_resolutions_as_int_list(res_path)
    print(res)