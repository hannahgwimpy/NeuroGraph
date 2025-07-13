"""Module for GNN and CTBN models."""

import torch
from torch_geometric.nn import GCNConv

class ProbabilisticGNN(torch.nn.Module):
    """A placeholder for the probabilistic GNN model."""
    def __init__(self):
        super(ProbabilisticGNN, self).__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = torch.dropout(x, p=0.5, train=self.training)
        x = self.conv2(x, edge_index)
        return torch.log_softmax(x, dim=1)
