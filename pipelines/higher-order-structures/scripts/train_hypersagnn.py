import os
import pandas as pd
import numpy as np
import sys

if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal

source_path = os.path.abspath("source/")
sys.path.append(source_path)
import utils as ut
import sagnn_utils as sag
from HyperSAGNN import HyperSAGNN


train_size = 0.6
emb_dim = 256
conv_dim = 128
order_threshold = 2
num_heads = 3
learning_rate = 0.001
weight_decay = 5e-4
weight_decay = 0.01
max_epoch = 20


if __name__ == "__main__":
    incidence_path = sys.argv[1]  
    outpath = sys.argv[2]

    print()
    print()
    print()
    print()
    print("########### WARNING: sampling columns ###########")
    columns = np.random.choice(range(100000), 100, replace=False)
    columns = [str(x) for x in columns]
    columns.insert(0, 'bin')
    
    # load data
    df = pd.read_csv(incidence_path, usecols=columns)
    df = df.set_index('bin')
    
    print(f"shape = {df.shape}")

    I = sag.prepare_training_data(df, order_threshold, train_size)
    print(I.shape)
    
#     incidence_matrix.to_csv(outpath, index=True)
    
    
    
    

    