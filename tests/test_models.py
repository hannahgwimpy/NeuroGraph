"""Tests for the models module."""

from neurograph.models import ProbabilisticGNN

def test_gnn_initialization():
    """Tests that the ProbabilisticGNN model can be initialized."""
    model = ProbabilisticGNN()
    assert model is not None
