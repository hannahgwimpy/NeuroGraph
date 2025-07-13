"""Module for data loading and processing."""

import json

def load_model_from_json(path):
    """Loads a model definition from a JSON file."""
    with open(path, 'r') as f:
        model_config = json.load(f)
    return model_config
