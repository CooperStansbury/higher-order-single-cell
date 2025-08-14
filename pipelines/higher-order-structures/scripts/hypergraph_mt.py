import os
import sys
import pandas as pd
import numpy as np
import glob
import time
import gget
import scipy
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from importlib import reload
import sys
sys.path.append('/home/cstansbu/git_repositories/Hypergraph-MT/code/HyMT')
sys.path.append('/home/cstansbu/git_repositories/Hypergraph-MT/code/')
import HyMT as hymt


def get_hyperedges(H, edge_weights=None, sample_size=1000, max_order=5):
    """
    Extracts hyperedges from an incidence matrix H.
    
    This function samples columns from the incidence matrix, identifies hyperedges 
    (sets of nodes connected by a hyperedge), and optionally filters them based on their order.
    
    Args:
      H: pandas DataFrame, the incidence matrix where rows represent nodes 
         and columns represent hyperedges.
      edge_weights: numpy array, optional weights for each hyperedge. If None, 
                    all hyperedges are assigned a weight of 1.
      sample_size: int, the number of columns (hyperedges) to sample from H.
      max_order: int, the maximum allowed hyperedge order (number of nodes in a hyperedge).
                 If None, no filtering is performed.
    
    Returns:
      tuple: (B, A, hyperedges, edge_idx)
        B: pandas DataFrame, the sampled incidence matrix.
        A: numpy array, hyperedge weights.
        hyperedges: list of tuples, each tuple representing a hyperedge with node indices.
        edge_idx: list of int, indices of the hyperedges that pass the filtering criteria.
    """
    B = H.sample(sample_size, axis=1)  # Randomly sample columns
    B = B[B.sum(axis=1) > 0]  # Keep rows with at least one positive entry

    if edge_weights is None:
        A = np.ones(sample_size)

    hyperedges = B.apply(lambda x: tuple(x[x > 0].index), axis=0).to_numpy()  

    # Hyperedge filtering
    orders = [len(e) for e in hyperedges]
    if max_order:
        edge_idx = [eid for eid, d in enumerate(orders) if 2 <= d <= max_order]
    else:
        edge_idx = [eid for eid, _ in enumerate(orders)]

    return B, A, hyperedges, edge_idx


def predict_hyperedge(hyperedge, u, w, index):
    """Calculates the probability of a hyperedge being non-zero.

    Args:
      hyperedge: The hyperedge.
      u: Membership matrix.
      w: Affinity matrix.
      index: Index for the DataFrame.

    Returns:
      A tuple containing the calculated value (M) and the probability.
    """
    u = pd.DataFrame(u, index=index)
    M = (np.prod(u.loc[np.array(hyperedge)], axis=0) * w[len(hyperedge) - 2]).sum()
    proba = 1.0 - np.exp(-M)
    return M, proba


if __name__ == "__main__":
    fpath = sys.argv[1] 
    K = int(sys.argv[2])
    u_outpath = sys.argv[3] 
    w_outpath = sys.argv[4] 
    training_outpath = sys.argv[5] 
    preds_outpath = sys.argv[6] 

    print(f"u_outpath: {u_outpath}")
    print(f"w_outpath: {w_outpath}")
    print(f"training_outpath: {training_outpath}")
    print(f"preds_outpath: {preds_outpath}")
    
    """ LOAD the hypergraph - always using the same one """
    H = pd.read_pickle(fpath)
    print(f"{H.shape=}")

    """ SAMPLE for testing """
    edge_weights = None
    sample_size = H.shape[1] # the WHOLE DATASET!
    max_order = 8
    
    B, A, hyperedges, edge_idx = get_hyperedges(
        H, 
        edge_weights=edge_weights,
        sample_size=sample_size,
        max_order=max_order,
    )
    
    print(f"Incidence matrix B shape: {B.shape}")
    print(f"Hyperedge weight vector A shape: {A.shape}")
    print(f"Number of hyperedges found: {len(hyperedges)}")
    print(f"Keeping {len(edge_idx)} out of {len(hyperedges)} hyperedges after filtering ({100*(len(edge_idx) / len(hyperedges)):.2f}%).")

    """ CONFIGURATION """
    conf_inf = {
        "seed": 10,
        "constraintU": False,
        "fix_communities": False,
        "fix_w": False,
        "gammaU": 0,
        "gammaW": 0,
        "initialize_u": None,  # Use None for null
        "initialize_w": None,  # Use None for null
        "out_inference": False,
        "plot_loglik": False,
    }

    """ TRAIN the model """
    num_realizations = 10
    
    model = hymt.model.HyMT(
        verbose=False,
        num_realizations=num_realizations,
    )
    
    u, w, maxL = model.fit(
        A[edge_idx], 
        hyperedges[edge_idx], 
        B.to_numpy()[:, edge_idx],
        K=K,
        **conf_inf,
    )

    print(f"max log-liklihood: {maxL}")
    
    """ COMPUTE predicted probabilities """
    preds = []
    for hyperedge in hyperedges[edge_idx]:
        p, M = predict_hyperedge(hyperedge, u, w, B.index)
        preds.append({
            'hyperedge' : "-".join(hyperedge),
            'p' : p,
            'M' : M,
        })

    preds = pd.DataFrame(preds)
    preds = preds.sort_values(by='p', ascending=False).reset_index(drop=True)

    """ STORE results """
    pd.DataFrame.sparse.from_spmatrix(
        csr_matrix(u),
        index=B.index,
        columns=[f"group_{x+1}" for x in range(K)],
    ).to_pickle(u_outpath)

    pd.DataFrame.sparse.from_spmatrix(
        csr_matrix(w),
        index=[f"degree_{x}" for x in range(w.shape[0])],
        columns=[f"group_{x+1}" for x in range(K)],
    ).to_pickle(w_outpath)

    model.train_info.to_parquet(
        training_outpath, 
        index=False,
    )

    preds.to_parquet(
        preds_outpath, 
        index=False,
    )



    
    

   
    
 

            

            
            

            
            
    
    
    
    
    
    
    
    
    
    
    