import pandas as pd
from sklearn import metrics
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
import os
import sys
import numpy as np

tool_path = "/home/cstansbu/miniconda3/lib/python3.9/site-packages/"
sys.path.append(tool_path)
import graph_tool.all as gt
print(f"{gt.__version__=}")


def learn_hyperedges_mcmc(g, niter=10):
    """
    Extracts hyperedges from a graph using MCMC and CliqueState.

    Args:
        g: A graph_tool.Graph object representing the graph.
        niter: Number of iterations for the MCMC sweep.

    Returns:
        A list of hyperedges, where each hyperedge is a list of nodes.
    """

    state = gt.CliqueState(g)
    state.mcmc_sweep(niter=niter)

    hyperedges = []
    for v in state.f.vertices():  
        if state.is_fac[v]:
            continue

        if state.x[v] > 0:
            hyperedge = list(state.c[v])
            hyperedges.append(hyperedge)

    return hyperedges


def create_graph_tools_from_adjacency(adjacency_matrix):
    """Creates a graph_tool Graph object from an adjacency matrix.

    Args:
        adjacency_matrix (np.ndarray): The input adjacency matrix.

    Returns:
        graph_tool.Graph: The generated Graph object.
    """
    nonzero_indices = np.nonzero(adjacency_matrix)
    edge_list = list(zip(*nonzero_indices)) 
    return gt.Graph(edge_list, directed=False) 