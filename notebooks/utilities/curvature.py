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



def compute_forman_curvature(
    VH: pd.DataFrame,
    v_weights: pd.Series | None = None,   # index = VH.index (vertices)
    e_weights: pd.Series | None = None,   # index = VH.columns (hyperedges)
    EF: pd.DataFrame | None = None,       # (hyperedges × faces) incidence, optional
    f_weights: pd.Series | None = None    # index = EF.columns (faces)
) -> pd.Series:
    """
    Forman curvature for each hyperedge e given:
      - VH: vertex×hyperedge (0/1 or bool) incidence matrix.
      - Optional vertex weights (default 1), hyperedge weights (default 1).
      - Optional faces via EF (hyperedge×face incidence) and face weights (default 1).

    Formula (edge-level):
        F(e) =  sum_{v<e}  w(v)/w(e)
              + sum_{f>e}  w(e)/w(f)
              - sum_{e' || e} [ sum_{v<e, v<e'}  w(v) / sqrt(w(e) w(e'))
                               + sum_{f>e, f>e'}  w(e) / w(f) ]

    Notes
    -----
    - v<e: vertices in e
    - f>e: (optional) faces containing e
    - e'||e: "parallel" edges (share a vertex or share a face, but NOT both when faces exist)
    - If EF is None, the face terms vanish and "parallel" reduces to edges sharing ≥1 vertex.
    """
    # Coerce to boolean
    VHb = VH.astype(bool)

    # Defaults: all weights = 1
    if v_weights is None:
        v_weights = pd.Series(1.0, index=VHb.index)
    else:
        v_weights = v_weights.reindex(VHb.index).fillna(1.0).astype(float)

    if e_weights is None:
        e_weights = pd.Series(1.0, index=VHb.columns)
    else:
        e_weights = e_weights.reindex(VHb.columns).fillna(1.0).astype(float)

    if EF is not None:
        EFb = EF.astype(bool).reindex(index=VHb.columns, fill_value=False)
        if f_weights is None:
            f_weights = pd.Series(1.0, index=EFb.columns)
        else:
            f_weights = f_weights.reindex(EFb.columns).fillna(1.0).astype(float)

    # Precompute vertex sets and (optional) face sets for each edge
    edge_vertices = {e: set(VHb.index[VHb[e].values]) for e in VHb.columns}
    if EF is not None:
        edge_faces = {e: set(EFb.columns[EFb.loc[e].values]) for e in VHb.columns}

    # Build adjacency of edges by shared vertices (Ev)
    # A_v[e] = set of edges sharing ≥1 vertex with e (excluding e)
    vertex_to_edges = {v: set(VHb.columns[VHb.loc[v].values]) for v in VHb.index}
    Av = {e: set() for e in VHb.columns}
    for v, edges_here in vertex_to_edges.items():
        for e in edges_here:
            Av[e].update(edges_here)
    for e in VHb.columns:
        Av[e].discard(e)

    # If faces exist, also build adjacency by shared faces (Af)
    if EF is not None:
        face_to_edges = {f: set(EFb.index[EFb[f].values]) for f in EFb.columns}
        Af = {e: set() for e in VHb.columns}
        for f, edges_here in face_to_edges.items():
            for e in edges_here:
                Af[e].update(edges_here)
        for e in VHb.columns:
            Af[e].discard(e)

    # Curvature accumulation
    F = pd.Series(0.0, index=VHb.columns)

    for e in VHb.columns:
        we = e_weights[e]

        # --- Down (vertex) term: sum_{v<e} w(v)/w(e)
        verts_e = edge_vertices[e]
        term_down = sum(v_weights[v] / we for v in verts_e)

        # --- Up (face) term: sum_{f>e} w(e)/w(f)  (0 if no faces)
        if EF is not None:
            faces_e = edge_faces[e]
            term_up = sum(we / f_weights[f] for f in faces_e)
        else:
            term_up = 0.0

        # --- Parallel edges and their contributions
        if EF is None:
            # No faces: "parallel" = share ≥1 vertex
            parallels = Av[e]
        else:
            # With faces: share a vertex XOR share a face
            parallels = (Av[e] ^ Af[e])  # symmetric difference = (share v or share f) but not both

        sum_parallel = 0.0
        for ep in parallels:
            wep = e_weights[ep]

            # Shared vertices term: sum_{v<e, v<e'}  w(v) / sqrt(w(e) w(e'))
            shared_vs = edge_vertices[e].intersection(edge_vertices[ep])
            term_parallel_down = sum(
                v_weights[v] / np.sqrt(we * wep) for v in shared_vs
            )

            # Shared faces term (only if EF): sum_{f>e, f>e'} w(e) / w(f)
            if EF is not None:
                shared_fs = edge_faces[e].intersection(edge_faces[ep])
                term_parallel_up = sum(we / f_weights[f] for f in shared_fs)
            else:
                term_parallel_up = 0.0

            sum_parallel += (term_parallel_down + term_parallel_up)

        F[e] = term_down + term_up - sum_parallel

    return F
