"""
Utility functions for the NeuroGraph project.
"""
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt

def config_to_graph(config):
    """Converts a model configuration dictionary into a PyTorch Geometric Data object.

    This function creates a graph representation of a CTBN ion channel model,
    where nodes are the channel states and edges represent possible transitions.

    Args:
        config (dict): A dictionary containing model parameters, loaded from JSON.

    Returns:
        torch_geometric.data.Data: A graph object ready for use with PyTorch Geometric.
    """
    num_states = config['num_states']
    nodes = torch.arange(num_states, dtype=torch.long)

    edge_list = []

    if num_states == 12:
        # 12-state model topology
        # Activation pathway: C0 <-> C1 <-> C2 <-> C3 <-> C4 <-> O
        for i in range(5):
            edge_list.append((i, i + 1))
            edge_list.append((i + 1, i))
        # Inactivation pathway: C_i <-> I_i
        for i in range(6):
            edge_list.append((i, i + 6))
            edge_list.append((i + 6, i))

    elif num_states == 24:
        # 24-state model topology (two 12-state copies + drug binding)
        # First copy (unbound)
        for i in range(5):
            edge_list.append((i, i + 1))
            edge_list.append((i + 1, i))
        for i in range(6):
            edge_list.append((i, i + 6))
            edge_list.append((i + 6, i))
        # Second copy (bound)
        for i in range(5):
            edge_list.append((i + 12, i + 1 + 12))
            edge_list.append((i + 1 + 12, i + 12))
        for i in range(6):
            edge_list.append((i + 12, i + 6 + 12))
            edge_list.append((i + 6 + 12, i + 12))
        # Drug binding/unbinding edges
        for i in range(12):
            edge_list.append((i, i + 12))
            edge_list.append((i + 12, i))

    else:
        raise ValueError(f"Unsupported number of states: {num_states}")

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Create the graph data object
    graph_data = Data(x=nodes.view(-1, 1).float(), edge_index=edge_index)
    graph_data.num_nodes = num_states

    return graph_data


def visualize_feature_importance(importance_scores, feature_names):
    """Visualizes feature importance scores."""
    fig, ax = plt.subplots()
    ax.barh(feature_names, importance_scores)
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance")
    plt.show()
