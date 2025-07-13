"""Utility functions for the NeuroGraph project."""

import matplotlib.pyplot as plt

def visualize_feature_importance(importance_scores, feature_names):
    """Visualizes feature importance scores."""
    fig, ax = plt.subplots()
    ax.barh(feature_names, importance_scores)
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance")
    plt.show()
