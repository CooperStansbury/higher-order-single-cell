import pandas as pd
import numpy as np
from collections import Counter, defaultdict


def nodes_neighbors(H: list[list[int]]) -> dict[int, set[int]]:
    """
    For each node, the set of all other nodes with which it co-occurs in any hyperedge.
    """
    v_neigh: dict[int, set[int]] = defaultdict(set)
    for edge in H:
        s = set(edge)
        for v in edge:
            v_neigh[v].update(s - {v})
    return dict(v_neigh)


def compute_hlrc(H: list[list[int]]) ->list[[float]]:
    """
    Compute the Hypergraph Lower Ricci Curvature (HLRC) for each hyperedge in a hypergraph.

    Parameters
    ----------
    H : list[list[int]]
        Hypergraph represented as a list of hyperedges, where each hyperedge is a list
        of integer node IDs. Duplicate nodes within a hyperedge are ignored.

    Returns
    -------
    list[Optional[float]]
        A list of curvature values, one per hyperedge. Returns `None` for a hyperedge if:
        - it contains 1 or fewer distinct nodes
        - any node in the hyperedge has zero neighbors in the hypergraph

    Method
    ------
    1. Construct a neighbor set for each node using `nodes_neighbors(H)`, where neighbors
       are all nodes that co-occur with the node in at least one hyperedge.
    2. For each hyperedge:
       - Determine the number of distinct nodes (`d_e`).
       - Compute each node's neighborhood size, as well as the maximum and minimum sizes.
       - Find the set of nodes that are neighbors of *every* node in the hyperedge (`common`).
       - Compute a harmonic term: sum of reciprocals of the neighborhood sizes.
       - Combine these into the HLRC score:

         HLRC(edge) = (Σ_{v∈edge} 1/|N(v)|) - 1
                      + (n_e + d_e/2 - 1) / max_size
                      + (n_e + d_e/2 - 1) / min_size

         where:
         - |N(v)| is the size of node v's neighborhood
         - n_e = number of common neighbors
         - d_e = number of distinct nodes in the edge

    Notes
    -----
    - Higher HLRC values indicate greater connectivity and neighborhood overlap within a hyperedge.
    - Sensitive to imbalance: extreme differences between max and min neighborhood sizes
      will affect the curvature.
    - Adapted from: https://github.com/shiyi-oo/hypergraph-lower-ricci-curvature/blob/main/code/src/hlrc.py
    """
    v_neigh = nodes_neighbors(H)
    
    hlrc: List[Optional[float]] = []
    for edge in H:
        d_e = len(set(edge))
        if d_e <= 1:
            hlrc.append(None)
            continue
        
        neigh_sizes = [len(v_neigh[v]) for v in edge]
        max_size, min_size = max(neigh_sizes), min(neigh_sizes)
        if max_size == 0 or min_size == 0:
            hlrc.append(None)
            continue
        
        common = set.intersection(*(v_neigh[v] for v in edge))
        n_e = len(common)
        sum_recip = sum(1 / s for s in neigh_sizes)
        
        e_hlrc = (
            sum_recip - 1
            + (n_e + d_e/2 - 1) / max_size
            + (n_e + d_e/2 - 1) / min_size
        )
        hlrc.append(e_hlrc)
    
    return hlrc