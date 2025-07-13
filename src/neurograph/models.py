"""Module for GNN and CTBN models."""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class NeuroGraphGNN(torch.nn.Module):
    """A Graph Neural Network to predict ion channel model parameters.

    This GNN takes a graph representing the states and transitions of an ion
    channel model and learns to predict the biophysical parameters that govern
    its dynamics.
    """
    def __init__(self, num_nodes, embedding_dim=32, hidden_dim=64, out_dim=18):
        """
        Args:
            num_nodes (int): The number of nodes (states) in the graph.
            embedding_dim (int): The dimensionality of the node embeddings.
            hidden_dim (int): The dimensionality of the hidden graph convolutional layers.
            out_dim (int): The number of output parameters to predict.
        """
        super(NeuroGraphGNN, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, embedding_dim)
        
        self.conv1 = GCNConv(embedding_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, data):
        """Defines the forward pass of the GNN.

        Args:
            data (torch_geometric.data.Data): The input graph data object.
                - data.x should contain the node indices.
                - data.edge_index should contain the graph connectivity.
                - data.batch is required for pooling over a batch of graphs.

        Returns:
            torch.Tensor: A tensor containing the predicted model parameters.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Get node embeddings from their indices
        x = self.embedding(x.squeeze().long())

        # 2. Apply graph convolutions
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # 3. Global pooling to get a graph-level representation
        graph_embedding = global_mean_pool(x, batch)

        # 4. Predict parameters from the graph embedding
        predicted_params = self.mlp(graph_embedding)

        return predicted_params

