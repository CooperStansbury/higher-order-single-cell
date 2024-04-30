import torch
import numpy as np
from node2vec import Node2Vec
import networkx as nx
import math

np.random.seed(0)


def create_hyperedge_index(incidence_matrix):
    row, col = torch.where(incidence_matrix.T)
    hyperedge_index = torch.cat((col.view(1, -1), row.view(1, -1)), dim=0)
    return hyperedge_index


def create_neg_incidence_matrix(incidence_matrix):
    incidence_matrix_neg = torch.zeros(incidence_matrix.shape)
    for i, edge in enumerate(incidence_matrix.T):
        nodes = torch.where(edge)[0]
        nodes_comp = torch.tensor(list(set(range(len(incidence_matrix))) - set(nodes.tolist())))
        edge_neg_l = torch.tensor(np.random.choice(nodes, math.floor(len(nodes) * 0.5), replace=False))
        edge_neg_r = torch.tensor(np.random.choice(nodes_comp, len(nodes) - math.floor(len(nodes) * 0.5), replace=False))
        edge_neg = torch.cat((edge_neg_l, edge_neg_r))
        incidence_matrix_neg[edge_neg, i] = 1
    return incidence_matrix_neg


def hyperlink_score_loss(y_pred, y):
    negative_score = torch.mean(y_pred[y == 0])
    logistic_loss = torch.log(1 + torch.exp(negative_score - y_pred[y == 1]))
    loss = torch.mean(logistic_loss)
    return loss


def create_label(incidence_matrix_pos, incidence_matrix_neg):
    y_pos = torch.ones(len(incidence_matrix_pos.T))
    y_neg = torch.zeros(len(incidence_matrix_neg.T))
    return torch.cat((y_pos, y_neg))


def train_test_split(incidence_matrix, y_label, train_size):
    size = len(y_label)
    shuffle_index = torch.randperm(size)
    incidence_matrix = incidence_matrix[:, shuffle_index]
    y_label = y_label[shuffle_index]
    split_index = round(train_size * size)
    return incidence_matrix[:, :split_index], y_label[:split_index], incidence_matrix[:, split_index:], y_label[split_index:]


def node2vec(incidence_matrix, emb_dim):
    size = len(incidence_matrix)
    hyperedge_index = create_hyperedge_index(incidence_matrix)
    node_set = hyperedge_index[0]
    b = hyperedge_index[1].int()
    edgelist = []
    for i in range(b[-1] + 1):
        nodes = node_set[b == i]
        num_nodes = len(nodes)
        adj_matrix = torch.triu(torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes))
        row, col = torch.where(adj_matrix)
        row, col = nodes[row], nodes[col]
        for j in range(len(row)):
            edgelist.append((int(row[j]), int(col[j])))
    G = nx.Graph()
    G.add_nodes_from(range(size))
    G.add_edges_from(edgelist)
    node2vec_emb = Node2Vec(G, dimensions=emb_dim, quiet=True)
    emb = node2vec_emb.fit()
    return torch.tensor(emb.wv.vectors)
