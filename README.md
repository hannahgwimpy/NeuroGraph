# NeuroGraph

A hybrid GNN framework combining CTBNs and graph-based learning to model channel kinetics with interpretable uncertainty and temporal reasoning.

## Features

- **Modular Ion Channel Models:** Pluggable CTBN models defined via JSON configurations (e.g., 12-state Kuo & Bean, 24-state drug-extended).
- **Probabilistic Framework:** Built on PyTorch, JAX, and Pyro/NumPyro for Bayesian inference and uncertainty quantification.
- **GNN Integration:** Designed to leverage PyTorch Geometric for graph-based analysis of channel state interactions.
- **Interpretability:** Planned integration with Captum and SHAP for model explainability.
- **Simulation Engine:** Core simulation logic powered by SciPy's Ordinary Differential Equation (ODE) solvers.

## Key Concepts Represented

- **Biophysics of ion channels**
- **Probabilistic modeling**
- **Graph learning**
- **Time-series uncertainty**

## Project Structure

```
neurograph/
├── data/                   # Model configurations (JSON) and experimental data
├── notebooks/              # Jupyter notebooks for exploration and analysis
├── scripts/                # Helper and training scripts
├── src/
│   └── neurograph/           # Source code
│       ├── ctbn.py           # CTBN model definitions
│       ├── data_loader.py    # Data loading utilities (placeholder)
│       ├── models.py         # GNN model definitions (placeholder)
│       ├── training.py       # Training pipelines (placeholder)
│       └── utils.py          # Utility functions (placeholder)
├── tests/                  # Unit and integration tests
├── .gitignore
├── pyproject.toml
├── README.md
├── requirements.txt
└── setup.py
```

## Installation

It is recommended to use a virtual environment to manage dependencies.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/hannahgwimpy/NeuroGraph.git
    cd NeuroGraph
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install the project in editable mode:**
    ```bash
    pip install -e .
    ```

## Getting Started

To see the modular CTBN system in action, run the exploration notebook:

1.  **Start JupyterLab:**
    ```bash
    jupyter lab
    ```

2.  **Open and run the notebook:**
    Navigate to `notebooks/01_initial_exploration.ipynb` and execute the cells. This will demonstrate how to load model configurations from the JSON files and instantiate the simulation models.
