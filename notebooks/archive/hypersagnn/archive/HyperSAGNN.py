import torch
import torch.nn as nn
import torch_geometric.nn as gnn


class HyperSAGNN(nn.Module):
    def __init__(self, input_dim, emb_dim, conv_dim, num_heads):
        super(HyperSAGNN, self).__init__()
        self.linear_encoder = nn.Linear(input_dim[0], emb_dim)
        self.graph_conv = gnn.TransformerConv(emb_dim, conv_dim, heads=num_heads, concat=False, root_weight=False)
        self.tanh = nn.Tanh()
        self.layer_norm_s = gnn.LayerNorm(conv_dim)
        self.layer_norm_d = gnn.LayerNorm(conv_dim)
        self.linear_static = nn.Linear(emb_dim, conv_dim, bias=False)
        self.linear = nn.Linear(conv_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.mean_pool = gnn.global_mean_pool

    def forward(self, feature, incidence_matrix):
        adj_matrix = feature @ feature.T - torch.diag(torch.sum(feature, dim=1))
        x = self.tanh(self.linear_encoder(adj_matrix))
        x, hyperedge_index = self.partition(x, incidence_matrix)
        edge_index, batch = self.expansion(hyperedge_index)
        x_d = self.layer_norm_d(self.tanh(self.graph_conv(x, edge_index)))
        x_s = self.layer_norm_s(self.tanh(self.linear_static(x)))
        p = self.sigmoid(self.linear((x_d - x_s) ** 2))
        return self.mean_pool(p, batch)

    @staticmethod
    def expansion(hyperedge_index):
        node_set = hyperedge_index[0]
        b = hyperedge_index[1].int()
        edge_index = torch.empty((2, 0), dtype=torch.int64)
        batch = torch.empty(len(node_set), dtype=torch.int64)
        for i in range(b[-1] + 1):
            nodes = node_set[b == i]
            batch[nodes.long()] = i
            num_nodes = len(nodes)
            adj_matrix = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
            row, col = torch.where(adj_matrix)
            row, col = nodes[row], nodes[col]
            edge = torch.cat((row.view(1, -1), col.view(1, -1)), 0)
            edge_index = torch.cat((edge_index, edge), dim=1)
        return edge_index, batch

    @staticmethod
    def partition(x, incidence_matrix):
	row, col = torch.where(incidence_matrix.T)
        hyperedge_index = torch.cat((col.view(1, -1), row.view(1, -1)), dim=0)
        node_set, sort_index = torch.sort(hyperedge_index[0])
        hyperedge_index[1] = hyperedge_index[1][sort_index]
        x = x[node_set.long(), :]
        hyperedge_index[0] = torch.arange(0, len(hyperedge_index[0]))
        index_set, sort_index = torch.sort(hyperedge_index[1])
        hyperedge_index[1] = index_set
        hyperedge_index[0] = hyperedge_index[0][sort_index]
        return x, hyperedge_index
