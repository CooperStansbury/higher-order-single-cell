import torch
import numpy as np
from node2vec import Node2Vec
import networkx as nx
import math
import torch
import pandas as pd
from sklearn import metrics
from scipy.io import loadmat
from os.path import exists
from tqdm import tqdm
import time

np.random.seed(0)

def predict(incidence_matrix, model):
    """Infers predictions from the model.

    Args:
        incidence_matrix (torch.Tensor): Hypergraph incidence matrix.
        model (torch.nn.Module): The PyTorch model.

    Returns:
        torch.Tensor: The predicted scores or probabilities.
    """
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        predictions = model(incidence_matrix, incidence_matrix) 
        return predictions 


def auprc(y_true, y_score):
    """Calculates the area under the precision-recall curve (AUPRC).

    Args:
        y_true (array-like): True binary labels.
        y_score (array-like): Target scores or probabilities.

    Returns:
        float: The AUPRC score.
    """
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
    return metrics.auc(recall, precision)


def train(feature, y, incidence_matrix, model, optimizer):
    """Trains the model for a single step.

    Args:
        feature (torch.Tensor): Input features.
        y (torch.Tensor): True labels.
        incidence_matrix (torch.Tensor): Hypergraph incidence matrix.
        model (torch.nn.Module): The PyTorch model.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
    """
    model.train()
    optimizer.zero_grad()
    y_pred = model(feature, incidence_matrix)
    loss = hyperlink_score_loss(y_pred, y) 
    loss.backward()
    optimizer.step()


def test(feature, y, incidence_matrix, model):
    """Evaluates the model's performance.

    Args:
        feature (torch.Tensor): Input features.
        y (torch.Tensor): True labels.
        incidence_matrix (torch.Tensor): Hypergraph incidence matrix.
        model (torch.nn.Module): The PyTorch model.

    Returns:
        tuple: A tuple containing (AUC, AUPRC, Accuracy, F1-score).
    """
    model.eval()
    with torch.no_grad():
        y_pred = model(feature, incidence_matrix).squeeze()
        fpr, tpr, thresholds = metrics.roc_curve(y, y_pred)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        label = (y_pred >= optimal_threshold).detach().numpy()

        auc = metrics.roc_auc_score(y, y_pred.detach().numpy())
        prc = auprc(y, y_pred.detach().numpy())
        acc = metrics.accuracy_score(y, label)
        f1score = metrics.f1_score(y, label)

    return auc, prc, acc, f1score 


def create_hyperedge_index(incidence_matrix):
    """Creates a hyperedge index representation from an incidence matrix.

    Args:
        incidence_matrix (torch.Tensor): The hypergraph incidence matrix.

    Returns:
        torch.Tensor: A 2xN tensor where N is the number of hyperedges. 
                      The first row contains node indices, and the second 
                      row contains hyperedge indices.
    """
    row, col = torch.where(incidence_matrix.T)
    hyperedge_index = torch.cat((col.view(1, -1), row.view(1, -1)), dim=0)
    return hyperedge_index


def create_neg_incidence_matrix(incidence_matrix):
    """Creates a negative incidence matrix by sampling negative hyperedges.

    Args:
        incidence_matrix (torch.Tensor): The original hypergraph incidence matrix.

    Returns:
        torch.Tensor: A negative incidence matrix of the same shape as the input.
    """
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
    """Calculates the hyperlink score loss for hypergraph link prediction.

    This loss function is designed to penalize false negatives more heavily 
    and is tailored towards hypergraph-based models.

    Args:
        y_pred (torch.Tensor): Predicted scores for each hyperedge.
        y (torch.Tensor): True labels (1 for positive hyperedges, 0 for negative).

    Returns:
        torch.Tensor: The computed hyperlink score loss.
    """
    negative_score = torch.mean(y_pred[y == 0])
    logistic_loss = torch.log(1 + torch.exp(negative_score - y_pred[y == 1]))
    loss = torch.mean(logistic_loss)
    return loss

def create_label(incidence_matrix_pos, incidence_matrix_neg):
    """Creates labels for positive and negative hyperedges.

    Args:
        incidence_matrix_pos (torch.Tensor): Incidence matrix for positive hyperedges.
        incidence_matrix_neg (torch.Tensor): Incidence matrix for negative hyperedges.

    Returns:
        torch.Tensor: Concatenated labels with 1 for positive and 0 for negative hyperedges.
    """
    y_pos = torch.ones(len(incidence_matrix_pos.T))
    y_neg = torch.zeros(len(incidence_matrix_neg.T))
    return torch.cat((y_pos, y_neg))

def train_test_split(incidence_matrix, y_label, train_size):
    """Splits data into training and testing sets.

    Args:
        incidence_matrix (torch.Tensor): The hypergraph incidence matrix.
        y_label (torch.Tensor): Labels for each hyperedge.
        train_size (float): The proportion of data to include in the training set (between 0 and 1).

    Returns:
        tuple: A tuple containing:
            * incidence_matrix_train (torch.Tensor): Incidence matrix for the training set.
            * y_label_train (torch.Tensor): Labels for the training set.
            * incidence_matrix_test (torch.Tensor): Incidence matrix for the testing set.
            * y_label_test (torch.Tensor): Labels for the testing set.
    """
    size = len(y_label)
    shuffle_index = torch.randperm(size)
    incidence_matrix = incidence_matrix[:, shuffle_index]
    y_label = y_label[shuffle_index]
    split_index = round(train_size * size)
    return incidence_matrix[:, :split_index], y_label[:split_index], incidence_matrix[:, split_index:], y_label[split_index:]

def node2vec(incidence_matrix, emb_dim):
    """Generates Node2Vec embeddings for nodes in a hypergraph.

    Args:
        incidence_matrix (torch.Tensor): The hypergraph incidence matrix.
        emb_dim (int): The dimension of the desired Node2Vec embeddings.

    Returns:
        torch.Tensor: A tensor containing the Node2Vec node embeddings.
    """
    size = len(incidence_matrix)
    hyperedge_index = create_hyperedge_index(incidence_matrix)

    # Construct edge list representation for Node2Vec
    edge_list = []
    for hyperedge_id in range(hyperedge_index.shape[1]):
        nodes_in_hyperedge = hyperedge_index[0, hyperedge_id]
        for i in range(len(nodes_in_hyperedge)):
            for j in range(i + 1, len(nodes_in_hyperedge)):  # Create pairwise edges
                edge_list.append((nodes_in_hyperedge[i], nodes_in_hyperedge[j])) 

    G = nx.Graph()
    G.add_nodes_from(range(size))
    G.add_edges_from(edge_list)

    node2vec_emb = Node2Vec(G, dimensions=emb_dim, quiet=True)
    embedding_model = node2vec_emb.fit()
    return torch.tensor(embedding_model.wv.vectors)
