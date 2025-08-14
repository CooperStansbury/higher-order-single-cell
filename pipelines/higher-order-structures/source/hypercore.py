import json
import itertools
import pandas as pd
import numpy as np
import networkx as nx
import pickle as pk
from itertools import compress
import sys
import copy
import xgi

def xgi_to_incidence(hypergraph):
    """Converts an XGI hypergraph to its corresponding incidence matrix.

    The incidence matrix is a representation where rows represent nodes and 
    columns represent hyperedges. A cell has value 1 if the node is part of
    the hyperedge, and 0 otherwise.

    Args:
        hypergraph: An XGI hypergraph object.

    Returns:
        pandas.DataFrame: The incidence matrix as a DataFrame.
    """

    nodes = np.array(sorted(hypergraph.nodes))
    incidence_matrix = np.array([
        np.isin(nodes, list(hyperedge)).astype(int)
        for hyperedge in hypergraph.edges.members()
    ]).T  

    return pd.DataFrame(incidence_matrix, index=nodes)


def prepare_for_hypercore_decomp(data, k=1):
    """Prepares interaction data for hypercore decomposition.

    Performs filtering, sorting, and duplicate removal to ensure data is suitable for 
    identifying hypercores with a minimum degree of 'k'.

    Args:
        data (list): A list of interactions, where each interaction is a list of nodes.
        k (int, optional): The minimum interaction size for inclusion. Defaults to 1.

    Returns:
        list: The processed list of unique, sorted interactions, ready for hypercore analysis.
    """

    # Filter out interactions smaller than 'k'
    filtered_data = [interaction for interaction in data if len(interaction) > k]

    # Sort each interaction's nodes
    for interaction in filtered_data:
        interaction.sort()

    # Sort the interactions themselves 
    filtered_data.sort()

    # Remove duplicate interactions
    unique_data = list(k for k, _ in itertools.groupby(filtered_data))

    # Sort by interaction length
    unique_data.sort(key=len)

    return unique_data



def build_xgi_hypergraph(data):
    """Builds a hypergraph using the xgi library.

    Args:
        data: The input data to build the hypergraph from.

    Returns:
        tuple: 
            - X (xgi.Hypergraph): The constructed hypergraph object.
            - nodes (list): List of nodes in the hypergraph.
            - edges (list): List of edges in the hypergraph.
    """
    X = xgi.Hypergraph(data)
    return X, list(X.nodes), list(X.edges)


def init_m_hypergraph(data, m):
    """Filters hyperedges by size and constructs a new hypergraph.

    Args:
        data: The input data for building the hypergraph (a list of sets/lists).
        m (int): The minimum edge size. Edges smaller than 'm' are excluded.

    Returns:
        tuple:
            - xgi.Hypergraph: The new hypergraph containing only edges of size 'm' or greater. 
            - nodes (list): List of nodes in the new hypergraph.
            - edges (list): List of edges in the new hypergraph.
    """
    m_hyperedges = [x for x in data if len(x) >= m]
    X, nodes, edges = build_xgi_hypergraph(m_hyperedges)
    return X, nodes, edges


def prune_edges_xgi(X, m):
    """Removes small hyperedges and duplicate hyperedges from a hypergraph.

    Args:
        X (xgi.Hypergraph): The input hypergraph.
        min_edge_size (int): The minimum size threshold for hyperedges. 
                             Hyperedges smaller than this size are removed.

    Returns:
        tuple:
            - xgi.Hypergraph: The modified hypergraph with small and duplicate edges removed.
            - list:  A list of the removed hyperedges.
    """
    edges = list(X.edges)
    sizes = [val for (edge, val) in X.edges.size] 
    edges_to_remove = [edges[i] for i in range(len(edges)) if sizes[i] < m ] 
    X.remove_edges_from(edges_to_remove)  
    
    # fully coincident hyperedges are removed
    unique_edges = [list(e) for e in X._edge.values()]
    unique_edges.sort()
    unique_edges = list(e for e, _ in itertools.groupby(unique_edges))  
    
    X, _, _ = build_xgi_hypergraph(unique_edges)
    return X, edges_to_remove
    
    
def filter_nodes_xgi(X, k):
    """Removes nodes from a hypergraph based on their degree.

    Args:
        X (xgi.Hypergraph): The input hypergraph.
        min_degree (int): The minimum degree threshold. Nodes with degrees 
                          less than 'min_degree' are removed.

    Returns:
        tuple:
            - xgi.Hypergraph: The modified hypergraph with low-degree nodes removed.
            - numpy.ndarray: An array of the removed nodes. 
    """
    node_degree = np.asarray(X.nodes.degree.aslist())
    node_idx = np.argwhere(node_degree < k).ravel()
    nodes_to_remove = np.asarray(X.nodes)[node_idx]
    X.remove_nodes_from(nodes_to_remove)
    return X, nodes_to_remove


def m_k_decomposition(data, m, k):
    """Calculates the (m, k)-core of a hypergraph.

    The (m, k)-core is a subgraph induced by repeatedly removing:
     * Nodes with degree less than k 
     * Hyperedges with fewer than m nodes (after node removals)

    Args:
        data: A list of hyperedges (each hyperedge represented as a set or list).
        m: Minimum hyperedge size for inclusion in the core.
        k: Minimum node degree for inclusion in the core.

    Returns:
        networkx.Hypergraph: The (m, k)-core subgraph.
    """
    # Consider only hyperedges of size >= m
    X, _, _ = init_m_hypergraph(data, m)  
     
    # Filter nodes and prune edges
    X, nodes_removed = filter_nodes_xgi(X, k)  
    X, edges_removed = prune_edges_xgi(X, m)  

    # Fix until satisfied
    while len(nodes_removed) > 0 or len(edges_removed) > 0:
        X, nodes_removed = filter_nodes_xgi(X, k)
        X, edges_removed = prune_edges_xgi(X, m)
    return X


def get_k_core(data, k_iter, k_step):
    """Calculates k-shell decomposition metrics for a hypergraph.

    The function iteratively removes nodes and hyperedges based on node degree
    and hyperedge size to reveal the k-shell structure of the hypergraph.

    Args:
        data: A list of hyperedges (each hyperedge represented as a set or list).
        k_iter: The maximum number of k-shell iterations.
        k_step: The step size for k values.

    Returns:
        m_k_core: A dictionary mapping (m, k) pairs to the corresponding (k, m)-core 
                  sub-hypergraph.
        k_shell_dict: A dictionary mapping nodes to their k-shell values at each order.
        k_max: An array indicating the maximum connectivity at each order.
    """
    n_hyperedges = len(data)
    max_order = len(data[-1])
    orders = np.asarray([len(x) for x in data])

    # Build the hypergraph
    X, nodes, edges = build_xgi_hypergraph(data)  # Assuming you have this function defined

    M = range(2, max_order + 1)
    K = range(0, k_iter, k_step)

    k_max = np.zeros(len(M))
    k_shell_dict = {node: np.zeros(len(M)) for node in nodes}
    m_k_core = {}

    # Iterate through each order
    for i, m in enumerate(M):
        D = np.zeros(len(K))

        # Consider only hyperedges of size >= m
        X, nodes, edges = init_m_hypergraph(data, m)  # Assuming you have this function

        for j, k in enumerate(K):
            previous_shell = nodes

            # Filter nodes and prune edges
            X, nodes_removed = filter_nodes_xgi(X, k)  # Assuming you have this function
            X, edges_removed = prune_edges_xgi(X, m)  # Assuming you have this function
            nodes = list(X.nodes)

            # Fix until satisfied
            while len(nodes_removed) > 0 or len(edges_removed) > 0:
                X, nodes_removed = filter_nodes_xgi(X, k)
                X, edges_removed = prune_edges_xgi(X, m)
                nodes = list(X.nodes)

            m_k_core[m, k] = X
            shell = list(nodes)
            k_shell = list(sorted(set(previous_shell) - set(shell)))

            for node in k_shell:
                k_shell_dict[node][i] = k - k_step

            D[j] = len(X.nodes)

            if j > 0 and D[j] == 0 and D[j - 1] != 0:
                k_max[i] = k - k_step  # maximum connectivity at order m
            if D[j] == 0:
                break  # stop when the (k,m)-core is empty

    return m_k_core, k_shell_dict, k_max


def get_hypercoreness(data, k_iter, k_step):
    """Calculates hypercoreness metrics for nodes in a hypergraph.

    The function first determines the k-shell structure of the hypergraph. Then,
    for each node, it computes two hypercoreness metrics:

    * **h_core:** The average k-shell value of a node across all hyperedge orders, 
        normalized by the maximum connectivity at each order.
    * **h_core_w:** A weighted average of the node's k-shell values, where weights 
        correspond to the relative frequency of hyperedges of different orders.

    Args:
        data: A list of hyperedges (each hyperedge represented as a set or list).
        k_iter: The maximum number of k-shell iterations.
        k_step: The step size for k values.

    Returns:
        A pandas DataFrame containing the following columns for each node:
            * **node:** The node identifier.
            * **h_core:** The hypercoreness of the node.
            * **h_core_w:** The weighted hypercoreness of the node.
    """
    max_order = len(data[-1])
    orders = [len(x) for x in data]
    _, k_shell_dict, k_max = get_k_core(data, k_iter, k_step)
    
    Psi = [] # distribution of hyperedges size
    for m in range(2, max_order + 1):
        Psi.append(orders.count(m) / len(orders))  
    Psi = np.array(Psi)
    
    res = []
    for node in k_shell_dict:
        row = {
            'node' : node,
            'h_core' : sum(np.array(k_shell_dict[node]) / np.array(k_max)),
            'h_core_w' : sum(Psi * np.array(k_shell_dict[node]) / np.array(k_max))
        }
        res.append(row)
        
    return pd.DataFrame(res)