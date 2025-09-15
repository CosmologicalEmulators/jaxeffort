"""Mock emulator data for testing."""

import numpy as np
import jax.numpy as jnp
import json
import tempfile
import os
from pathlib import Path


def create_mock_nn_dict():
    """Create a mock neural network configuration dictionary."""
    return {
        "n_input_features": 8,
        "n_output_features": 100,
        "n_hidden_layers": 2,
        "layers": {
            "layer_1": {
                "n_neurons": 50,
                "activation_function": "tanh"
            },
            "layer_2": {
                "n_neurons": 30,
                "activation_function": "relu"
            }
        },
        "emulator_description": {
            "name": "Mock Emulator",
            "version": "1.0",
            "type": "test"
        }
    }


def create_mock_weights(nn_dict):
    """Create mock weights matching the neural network structure."""
    # Calculate total number of weights needed
    total_weights = 0

    # Input to first hidden layer
    total_weights += nn_dict["n_input_features"] * nn_dict["layers"]["layer_1"]["n_neurons"]
    total_weights += nn_dict["layers"]["layer_1"]["n_neurons"]  # biases

    # First hidden to second hidden
    total_weights += nn_dict["layers"]["layer_1"]["n_neurons"] * nn_dict["layers"]["layer_2"]["n_neurons"]
    total_weights += nn_dict["layers"]["layer_2"]["n_neurons"]  # biases

    # Second hidden to output
    total_weights += nn_dict["layers"]["layer_2"]["n_neurons"] * nn_dict["n_output_features"]
    total_weights += nn_dict["n_output_features"]  # biases

    # Generate random weights
    return np.random.randn(total_weights) * 0.1


def create_mock_minmax(n_features):
    """Create mock min-max normalization arrays."""
    # Format: [min, max] for each feature
    return np.column_stack([
        np.zeros(n_features),  # min values
        np.ones(n_features)    # max values
    ])


def create_mock_k_grid(n_k=50):
    """Create mock k-grid for power spectrum."""
    return np.logspace(-3, 0, n_k)  # k from 0.001 to 1 h/Mpc


def create_mock_postprocessing_file():
    """Create a mock postprocessing Python file content."""
    return '''"""Mock postprocessing for testing."""
import jax.numpy as jnp

def postprocessing(input_params, output, D, emulator):
    """Mock postprocessing function."""
    # Simple passthrough for testing
    return output * D**2
'''


def create_mock_bias_contraction_file():
    """Create a mock bias contraction Python file content."""
    return '''"""Mock bias contraction for testing."""
import jax.numpy as jnp

def BiasContraction(biases, stacked_array):
    """Mock bias contraction function."""
    # Simple linear combination for testing
    # biases: [b1, b2, bs2, b3nl]
    # For simplicity, just use b1 scaling
    b1 = biases[0]
    return stacked_array * b1**2
'''


def create_mock_emulator_directory(base_path=None):
    """Create a complete mock emulator directory structure with all necessary files."""
    if base_path is None:
        base_path = tempfile.mkdtemp(prefix="mock_emulator_")
    else:
        os.makedirs(base_path, exist_ok=True)

    nn_dict = create_mock_nn_dict()

    # Create component directories
    for component in ["11", "loop", "ct"]:
        comp_path = Path(base_path) / component
        comp_path.mkdir(exist_ok=True)

        # Save nn_setup.json
        with open(comp_path / "nn_setup.json", "w") as f:
            json.dump(nn_dict, f)

        # Save weights
        weights = create_mock_weights(nn_dict)
        np.save(comp_path / "weights.npy", weights)

        # Save normalization arrays
        in_minmax = create_mock_minmax(nn_dict["n_input_features"])
        out_minmax = create_mock_minmax(nn_dict["n_output_features"])
        np.save(comp_path / "inminmax.npy", in_minmax)
        np.save(comp_path / "outminmax.npy", out_minmax)

        # Save k-grid
        k_grid = create_mock_k_grid()
        np.save(comp_path / "k.npy", k_grid)

        # Save postprocessing.py
        with open(comp_path / "postprocessing.py", "w") as f:
            f.write(create_mock_postprocessing_file())

    # Save multipole-level bias contraction
    with open(Path(base_path) / "biascontraction.py", "w") as f:
        f.write(create_mock_bias_contraction_file())

    return base_path


def create_mock_noise_emulator_directory(base_path=None):
    """Create a mock emulator directory with noise component."""
    if base_path is None:
        base_path = tempfile.mkdtemp(prefix="mock_noise_emulator_")

    # First create the standard multipole emulator
    create_mock_emulator_directory(base_path)

    # Add noise component
    nn_dict = create_mock_nn_dict()
    noise_path = Path(base_path) / "st"
    noise_path.mkdir(exist_ok=True)

    # Save nn_setup.json for noise
    with open(noise_path / "nn_setup.json", "w") as f:
        json.dump(nn_dict, f)

    # Save weights
    weights = create_mock_weights(nn_dict)
    np.save(noise_path / "weights.npy", weights)

    # Save normalization arrays
    in_minmax = create_mock_minmax(nn_dict["n_input_features"])
    out_minmax = create_mock_minmax(nn_dict["n_output_features"])
    np.save(noise_path / "inminmax.npy", in_minmax)
    np.save(noise_path / "outminmax.npy", out_minmax)

    # Save k-grid
    k_grid = create_mock_k_grid()
    np.save(noise_path / "k.npy", k_grid)

    # Save postprocessing.py
    with open(noise_path / "postprocessing.py", "w") as f:
        f.write(create_mock_postprocessing_file())

    return base_path