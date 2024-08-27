import sys
import os
import pandas as pd
import numpy as np
import random
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from importlib import reload
import matplotlib.pyplot as plt
from matplotlib import patches
import seaborn as sns
import networkx as nx
from scipy import sparse
import hypernetx as hnx
import math
from itertools import product
from itertools import combinations
from itertools import groupby
from collections import defaultdict
from collections import Counter
import warnings
import time
import stats
from scipy.stats import zscore
from scipy.stats import qmc
warnings.simplefilter("ignore", category=RuntimeWarning)

source_path = os.path.abspath("../../source/")
sys.path.insert(0, source_path)
print(source_path)
import utils as ut
import plotting as plt2
import hypercore as hc
import matrix as matrix
import centrality as central

# hyperlocal imports
import core_utils
import nb_utils as nb


def nl_centrality_func(B, 
                       f=lambda x: x,
                       g=lambda x: x,
                       phi=lambda x: x,
                       psi=lambda x: x,
                       maxiter=1000, 
                       tol=1e-6, 
                       edge_weights=None,
                       node_weights=None,
                       mynorm=lambda x: np.linalg.norm(x, 1), 
                       verbose=False):
    """
    Computes centrality measures using SparseArrays and iterative algorithm.

    Args:
        B: Sparse adjacency matrix.
        functions: four functions
        maxiter: Maximum number of iterations.
        tol: Tolerance for convergence.
        edge_weights: Optional weights for edges (defaults to all 1).
        node_weights: Optional weights for nodes (defaults to all 1).
        mynorm: Norm function to use (defaults to L1 norm).
        verbose: If True, print progress messages during iterations.

    Returns:
        Two NumPy arrays: node centrality scores and edge centrality scores.
    """
    B = sp.csr_matrix(B)
    n, m = B.shape

    x0 = np.ones((n, 1)) / n
    y0 = np.ones((m, 1)) / m

    if edge_weights is None:
        edge_weights = np.ones((m, 1))
    if node_weights is None:
        node_weights = np.ones((n, 1))
    W = sp.diags(edge_weights.flatten())
    N = sp.diags(node_weights.flatten())

    for it in range(1, maxiter + 1):
        if verbose and (it < 10 or it % 30 == 0):
            print(f"{it} ...")
            
        u = np.sqrt(x0 * g(B @ W @ f(y0)))
        v = np.sqrt(y0 * psi(B.T @ N @ phi(x0)))
        x = u / mynorm(u)
        y = v / mynorm(v)

        check = mynorm(x - x0) + mynorm(y - y0)
        if check < tol:
            if verbose:
                print(f"{it} ===")
            return x.flatten(), y.flatten()

        x0 = x.copy()
        y0 = y.copy()

    if verbose:
        print(f"Warning: Centrality did not reach tol = {tol} in {maxiter} iterations\n------- Relative error reached so far = {check}")
    return x0.flatten(), y0.flatten()


def nonlinear_eigenvector_centrality(B, 
                                     function='linear',
                                     maxiter=1000, 
                                     tol=1e-6, 
                                     edge_weights=None,
                                     node_weights=None,
                                     mynorm=lambda x: np.linalg.norm(x, 1), 
                                     verbose=False):
    """
    Computes centrality measures using SparseArrays and iterative algorithm.

    Args:
        B: Sparse adjacency matrix.
        function: Key to select functions  (e.g., "linear", "log-exp", "max").
        maxiter: Maximum number of iterations.
        tol: Tolerance for convergence.
        edge_weights: Optional weights for edges (defaults to all 1).
        node_weights: Optional weights for nodes (defaults to all 1).
        mynorm: Norm function to use (defaults to L1 norm).
        verbose: If True, print progress messages during iterations.

    Returns:
        Two NumPy arrays: node centrality scores and edge centrality scores.
    """

    mappings = {
        "linear": (lambda x: x, lambda x: x, lambda x: x, lambda x: x),
        "log-exp": (lambda x: x, lambda x: np.power(x, 1/10), lambda x: np.log(x), lambda x: np.exp(x)),
        "max": (lambda x: x, lambda x: np.power(x, 1/5), lambda x: np.power(x, 15), lambda x: np.power(x, 1/15))
    }

    f, g, phi, psi = mappings[function]

    B = sp.csr_matrix(B)
    n, m = B.shape

    x0 = np.ones((n, 1)) / n
    y0 = np.ones((m, 1)) / m

    if edge_weights is None:
        edge_weights = np.ones((m, 1))
    if node_weights is None:
        node_weights = np.ones((n, 1))
    W = sp.diags(edge_weights.flatten())
    N = sp.diags(node_weights.flatten())

    for it in range(1, maxiter + 1):
        if verbose and (it < 10 or it % 30 == 0):
            print(f"{it} ...")
            
        u = np.sqrt(x0 * g(B @ W @ f(y0)))
        v = np.sqrt(y0 * psi(B.T @ N @ phi(x0)))
        x = u / mynorm(u)
        y = v / mynorm(v)

        check = mynorm(x - x0) + mynorm(y - y0)
        if check < tol:
            if verbose:
                print(f"{it} ===")
            return x.flatten(), y.flatten()

        x0 = x.copy()
        y0 = y.copy()

    if verbose:
        print(f"Warning: Centrality did not reach tol = {tol} in {maxiter} iterations\n------- Relative error reached so far = {check}")
    return x0.flatten(), y0.flatten()


def generate_core_periphery_hypergraph(num_core_nodes, num_periphery_nodes,
                                      edge_probability_core, edge_probability_periphery,
                                      avg_edge_size, core_periphery_probability):
    """
    This function generates a random hypergraph with a core-periphery structure and creates its binary incidence matrix.

    Args:
      num_core_nodes: Number of nodes in the core.
      num_periphery_nodes: Number of nodes in the periphery.
      edge_probability_core: Probability of an edge forming between core nodes.
      edge_probability_periphery: Probability of an edge forming between periphery nodes.
      avg_edge_size: Average number of nodes per edge.
      core_periphery_probability: Probability of an edge forming between a core node and a periphery node.

    Returns:
      A tuple containing four elements:
          * core_nodes: List of core nodes.
          * periphery_nodes: List of periphery nodes.
          * edges: List of edges.
          * incidence_matrix: Binary incidence matrix as a NumPy array.
    """

    # Define core and periphery nodes
    core_nodes = list(range(num_core_nodes))
    periphery_nodes = list(range(num_core_nodes, num_core_nodes + num_periphery_nodes))

    # Get total number of nodes
    total_nodes = len(core_nodes) + len(periphery_nodes)

    # Generate edges
    edges = []
    for _ in range(int(len(core_nodes) * edge_probability_core)):
        # Sample core nodes for an edge
        edge = random.sample(core_nodes, k=int(avg_edge_size))
        edges.append(edge)

    for _ in range(int(len(periphery_nodes) * edge_probability_periphery)):
        # Sample periphery nodes for an edge
        edge = random.sample(periphery_nodes, k=int(avg_edge_size))
        edges.append(edge)

    # Add edges between core and periphery nodes
    for _ in range(int(num_core_nodes * num_periphery_nodes * core_periphery_probability)):
        # Sample a core node and a periphery node
        core_node = random.choice(core_nodes)
        periphery_node = random.choice(periphery_nodes)
        # Create an edge with the core and periphery node
        edge = [core_node, periphery_node]
        # Optionally, you can sample additional nodes for the edge
        if avg_edge_size > 2:
            additional_nodes = random.sample(core_nodes + periphery_nodes, k=int(avg_edge_size) - 2)
            edge.extend(additional_nodes)
            
        edges.append(edge)

    # Create empty matrix
    incidence_matrix = np.zeros((total_nodes, len(edges)), dtype=int)

    # Fill the matrix with 1s for corresponding nodes in each edge
    for i, edge in enumerate(edges):
        for node in edge:
            incidence_matrix[node, i] = 1

    return core_nodes, periphery_nodes, edges, incidence_matrix
    

def get_core(H, outlier_indices, function='log-exp', q=0.75, maxiter=10000):
    """
    Calculates nonlinear eigenvector centrality and related metrics.

    Args:
        H: The adjacency matrix (pandas DataFrame).
        outlier_indices: List of indices to drop from H.
        function: Nonlinear function for centrality calculation (default: 'log-exp').
        q: Quantile threshold for defining 'core' nodes (default: 0.75).
        maxiter: Maximum iterations for centrality calculation.

    Returns:
        pandas.DataFrame: A DataFrame with node information and centrality metrics.
    """
    
    # Remove outliers and zero-sum columns (without explicit checks)
    Hhat = H.drop(outlier_indices).loc[:, (H.sum(axis=0) != 0)]

    # Calculate centrality using NetworkX
    nodes_cent, _ = central.nonlinear_eigenvector_centrality(
        Hhat, 
        function=function, 
        maxiter=maxiter,
    )

    # Create DataFrame with results
    nodes = pd.DataFrame({
        'local_bin': Hhat.index,
        'node_centrality': nodes_cent,
        'zscores': zscore(nodes_cent),
        'node_centrality_norm': nb.min_max(nodes_cent),  # Assuming you have ut.min_max elsewhere
        'core' : nodes_cent >= np.quantile(nodes_cent, q),
    })

    return nodes





def edges_by_centrality(G, centrality_dict):
    """
    Computes the edges of a graph based on their pseudo centrality and returns them sorted in descending order of centrality.

    This function calculates a pseudo centrality score for each edge in the graph by summing the centrality values 
    of its two endpoints. The edges are then sorted by this pseudo centrality score in descending order.

    Parameters:
    - G (iterable of tuples): An iterable containing edges of the graph, where each edge is represented as a tuple 
      of two nodes (e.g., (node1, node2)).
    - centrality_dict (dict): A dictionary mapping nodes to their centrality values. Centrality values should be 
      numerical. If a node is not present in the dictionary, a default value of 0 is used.

    Returns:
    - list: A list of edges sorted in descending order of their pseudo centrality scores. The pseudo centrality score 
      for each edge is computed as the sum of the centrality values of its two endpoint nodes.

    Example:
    >>> G = [(1, 2), (2, 3), (3, 4)]
    >>> centrality_dict = {1: 0.5, 2: 1.2, 3: 0.9, 4: 0.3}
    >>> edges_by_centrality(G, centrality_dict)
    [(2, 3), (1, 2), (3, 4)]
    """
    # Convert the centrality_dict to a function for fast lookups
    get_centrality = centrality_dict.get
    mapping = {}

    # Use dictionary comprehension to build the mapping
    for e in G:
        # Compute the pseudo edge centrality by summing up the centralities of each node in the edge
        pseudo_edge_centrality = sum(get_centrality(num, 0) for num in e)
        mapping[e] = pseudo_edge_centrality

    sorted_keys = [key for key, value in sorted(mapping.items(), key=lambda item: item[1], reverse=True)]
    return sorted_keys



#Runtime O(n*k)
def find_all_mappings(dictionary, integer_list):
    # List comprehension to generate all possible mappings
    dict_values = list(dictionary.values())+integer_list
    possible_swaps = [(integer, element) for integer in integer_list for element in dict_values]
    return possible_swaps

def map_tuple_to_tuple(original_tuple, mapping_dict):
    """
    Maps elements of a tuple to new values based on a mapping dictionary.

    This function takes a tuple and a dictionary where the dictionary keys represent
    the elements of the tuple to be mapped to new values. If an element in the tuple
    is found in the dictionary, it is replaced with the corresponding value from the
    dictionary. If an element is not found in the dictionary, it remains unchanged.

    Args:
        original_tuple (tuple): The tuple whose elements are to be mapped.
        mapping_dict (dict): A dictionary where keys are tuple elements and values are
                             the new values to map to. If a key is not present in the
                             dictionary, the original element is used.

    Returns:
        tuple: A new tuple with elements transformed according to the mapping dictionary.

    Example:
        >>> original_tuple = (1, 2, 3, 4)
        >>> mapping_dict = {1: 'a', 2: 'b', 4: 'd'}
        >>> map_tuple_to_tuple(original_tuple, mapping_dict)
        ('a', 'b', 3, 'd')
    """
    mapped_tuple = tuple(mapping_dict.get(item, item) for item in original_tuple)
    return mapped_tuple



def create_incidence_matrix_dataframe(hyperedges):
    # Step 1: Extract all unique vertices and hyperedges
    unique_vertices = sorted(set(v for hyperedge in hyperedges for v in hyperedge))
    unique_hyperedges = sorted(hyperedges)
    
    # Create a mapping from vertices to rows
    vertex_index = {v: i for i, v in enumerate(unique_vertices)}
    
    # Create a mapping from hyperedges to columns
    hyperedge_index = {he: j for j, he in enumerate(unique_hyperedges)}
    
    # Initialize the incidence matrix
    incidence_matrix = np.zeros((len(unique_vertices), len(unique_hyperedges)), dtype=int)
    
    # Fill the incidence matrix
    for hyperedge in unique_hyperedges:
        col = hyperedge_index[hyperedge]
        for vertex in hyperedge:
            row = vertex_index[vertex]
            incidence_matrix[row, col] = 1
    
    # Create a DataFrame from the incidence matrix
    df = pd.DataFrame(incidence_matrix, index=unique_vertices, columns=[str(he) for he in unique_hyperedges])
    
    return df


def to_hypernet(H):
    """A function to convert a dataframe into a 
    hypernet hypergrapgh"""
    
    iteractions = {}
    
    for idx, row in H.T.iterrows():
        iteractions[idx] = tuple(row[row == 1].index)

    hx = hnx.Hypergraph(iteractions)
    return hx
    


def node_color(v):
    if v in core_nodes:
        return 'r'
    else:
        return 'b'

def incidence_matrix_to_hyperedges(matrix):
    # Convert the incidence matrix to a numpy array for easier manipulation
    #matrix = np.array(incidence_matrix)
    row_sums = np.sum(matrix, axis=1)
    
    # Get the indices that would sort the row sums in descending order
    sorted_indices = np.argsort(row_sums)[::-1]
    
    # Reorder rows according to the sorted indices
    reordered_matrix = matrix[sorted_indices]
    # List to hold the hyperedges as tuples of vertices
    hyperedges = set()
    
    # Number of vertices and hyperedges
    num_vertices, num_hyperedges = reordered_matrix.shape
    
    # Loop through each hyperedge (column)
    for hyperedge_index in range(num_hyperedges):
        # Find vertices incident to this hyperedge
        vertices = np.where(matrix[:, hyperedge_index] == 1)[0]
        
        # Create a sorted tuple of vertices for the hyperedge
        if len(vertices) > 0:
            hyperedge = tuple(sorted(vertices))
            hyperedges.add(hyperedge)
    
    return hyperedges


def reduce_tuples(tuples_set):
    # Step 1: Extract all unique numbers
    unique_numbers = set()
    for tpl in tuples_set:
        unique_numbers.update(tpl)
    
    # Step 2: Create a mapping from original numbers to smallest integers
    sorted_numbers = sorted(unique_numbers)
    number_mapping = {num: idx for idx, num in enumerate(sorted_numbers)}
    
    # Step 3: Apply the mapping to each tuple
    reduced_tuples = {tuple(number_mapping[num] for num in tpl) for tpl in tuples_set}
    
    return reduced_tuples



def backtrack(current_mapping, remaining_keys, available_values, mappings, valid_mappings):
    """
    A helper function to perform backtracking to generate distinct mappings.

    Args:
        current_mapping (dict): The current state of the mapping being built.
        remaining_keys (list): The keys that still need to be mapped.
        available_values (list): The values available for mapping.
        mappings (dict): The mapping of keys to possible values.
        valid_mappings (list): The list to store valid mappings.
    """
    if not remaining_keys:
        valid_mappings.append(current_mapping.copy())
        return
    
    key = remaining_keys[0]
    for value in available_values:
        if value not in current_mapping.values() and value in mappings[key]:
            current_mapping[key] = value
            next_remaining_keys = remaining_keys[1:]
            backtrack(current_mapping, next_remaining_keys, available_values, mappings, valid_mappings)
            del current_mapping[key]  # Undo the choice

def generate_distinct_mappings(pairs):
    """
    Generates all distinct mappings for a given list of key-value pairs.

    Args:
        pairs (list of tuples): A list of tuples where each tuple contains a key and a value.

    Returns:
        list: A list of valid mappings where each mapping is a dictionary.
    """
    
    # Step 1: Group tuples by their first entry
    mappings = {}
    all_values = set()
    for key, value in pairs:
        if key not in mappings:
            mappings[key] = []
        mappings[key].append(value)
        all_values.add(value)
    
    # Step 2: Generate a list of keys and corresponding values
    keys = list(mappings.keys())
    values = list(all_values)
    
    # Step 3: Initialize valid_mappings and call backtrack
    valid_mappings = []
    backtrack({}, keys, values, mappings, valid_mappings)
    
    return valid_mappings

    
def map_tuple_and_check_for_repeats(missing_keys, dicts_list, tuples, mapping_dict):
    """
    Maps tuples using a dictionary and checks for duplicates in the mapped tuples.

    Args:
        tuples (list of tuples): The list of tuples to be mapped.
        mapping_dict (dict): A dictionary where keys are tuple elements and values
                             are the new values to map to.

    Returns:
        list of tuples: Mapped tuples.
        list of tuples: Tuples that have duplicates after mapping.
        dict: Mapping causing duplication, if any.
    """
    mapped_tuples = []
    problematic_mapping = {}
    for original_tuple in tuples:
        # Map the tuple
        mapped_tuple = tuple(mapping_dict.get(item, item) for item in original_tuple)
        mapped_tuples.append(mapped_tuple)
        # Check for duplicates
        if len(mapped_tuple) != len(set(mapped_tuple)):
            # Identify which mapping caused the problem
            for item in original_tuple:
                if item not in missing_keys:
                    continue
                mapped_item = mapping_dict.get(item, item)
                if list(mapped_tuple).count(mapped_item) > 1:
                    problematic_mapping[item] = mapped_item
            break
    if problematic_mapping:
        print('These are the mappings that are causing an error', problematic_mapping)
        for key, value in problematic_mapping.items():
            if key not in missing_keys:
                continue
            dicts_list = filter_dicts_mapping(dicts_list, key, value)
    return dicts_list


def filter_dicts_mapping(dicts_list, key, value):
    """
    Filters out dictionaries from the list that contain a specific key-value mapping.

    Args:
        dicts_list (list of dict): The list of dictionaries to be filtered.
        key (any): The key to check in each dictionary.
        value (any): The value associated with the key to exclude.

    Returns:
        list of dict: The filtered list of dictionaries that do not contain the specified key-value mapping.
    """
    return [d for d in dicts_list if d.get(key) != value]




def edge_core_alg(G):
    print('this is the OG graph: ', G)
    #Put the first edge in the core (requires preprocessing to ensure the first edge should indeed be in the core (biggest edge?))
    iterator = iter(G)
    C = next(iterator)
    mapping_dictionary = {x: x for x in C}
    C = {C}
    
    #For each edge in the graph, try to map any nodes not included in the mapping to the original graph
    for e in G:
        #Add the edge into the core
        print(e)
        new_e = map_tuple_to_tuple(e, mapping_dictionary)
        C.add(new_e)
        
        #Find all the nodes of the core
        keys_set = set(mapping_dictionary.values())
        
        #Find all the nodes in the edge that is not in the core
        missing_keys = [x for x in new_e if x not in keys_set]
        
        if missing_keys:
            

            #VERSION 2 FAILED
            #Find all possible simultaneous mappings
            possible_maps = generate_distinct_mappings(find_all_mappings(mapping_dictionary, missing_keys))
            #For each map
            #print(f'ALL POSS MAPPINGS: {possible_maps}')
            print('these are the possible maps: ', possible_maps)
            
           '''
           #VERSION 2
            while len(possible_maps) > 0:
                print('Possible mappings to check', len(possible_maps))
                map = possible_maps[0]
                #create the true mapping and test the validity of the map
                combined_mapping = {**mapping_dictionary, **map}
                #print(f'This is the proposed map: {combined_mapping}')
                new_possible_maps = map_tuple_and_check_for_repeats(missing_keys, possible_maps, G, map)
                if len(new_possible_maps) == len(possible_maps):
                    mapping_dictionary = combined_mapping
                    print(f'found the map: {map}')
                    break
                possible_maps = new_possible_maps
            '''

            '''
            #VERSION 1
            #Find all possible simultaneous mappings
            #possible_maps = generate_distinct_mappings(find_all_mappings(mapping_dictionary, missing_keys))
            #For each map
            #print(f'ALL POSS MAPPINGS: {possible_maps}')
            
            for map in possible_maps:
                
                #create the true mapping and test the validity of the map
                combined_mapping = {**mapping_dictionary, **map}
                #print(f'This is the first proposed map: {combined_mapping}')
                
                #Map graph using the proposed homomorphism
                test_graph = [tuple(sorted(map_tuple_to_tuple(test_edges, combined_mapping))) for test_edges in G]
                #print(f'This is the mapped sets of edges: {test_graph}')
                #print(f'This is the original set of edges: {G}')
                edge_lengths = [len(set(tpl)) for tpl in G]
                post_edge_lengths = [len(set(tpl)) for tpl in test_graph]
                
                if edge_lengths == post_edge_lengths:
                    #print(f'found the map')
                    
                    mapping_dictionary = combined_mapping
                    break_condition = True
                    break
                append_dictionary = {x: x for x in missing_keys}
                mapping_dictionary = {**mapping_dictionary, **append_dictionary}
            '''
        
        G = set(map_tuple_to_tuple(test_edges, mapping_dictionary) for test_edges in G)
    print(f'This is how many nodes are in the core {len(set(mapping_dictionary.values()))}')
    return mapping_dictionary


def betterSolve(G):
    print(G)
    map = edge_core_alg(G)
    new_set = set(tuple(sorted(map_tuple_to_tuple(e, map))) for e in G)
    print(new_set)
    df_imatrix = create_incidence_matrix_dataframe(new_set)
    
    
    hx = to_hypernet(df_imatrix)
    '''
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['figure.figsize'] = (6, 6)
    hnx.drawing.draw(hx, 
                     with_node_counts=False, 
                     with_edge_counts=False,
                     with_edge_labels=False, 
                     with_node_labels=False,
                     nodes_kwargs={'color': node_color, 'edgecolor': 'k'},
                     edges_kwargs={'edgecolors': 'grey'},
                     layout_kwargs={'seed': 39})
    '''
    return hx, set(map.values())