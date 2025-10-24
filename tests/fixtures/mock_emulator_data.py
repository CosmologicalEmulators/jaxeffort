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
        "n_input_features": 9,  # 9 cosmology parameters
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
            "type": "test",
            "parameters": "z, ln10^10 As, ns, H0, omega_b, omega_c, Mnu, w0, wa"
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


def create_mock_postprocessing_file(n_components):
    """Create a mock postprocessing Python file content."""
    return f'''"""Mock postprocessing for testing."""
import jax.numpy as jnp

def postprocessing(input_params, output, D, emulator):
    """Mock postprocessing function."""
    # Apply D^2 scaling
    # Return flat array - MLP.get_component will reshape it
    return output * D**2
'''


def create_mock_bias_combination_file():
    """Create a mock bias combination Python file content (PyBird-style with 11 params)."""
    return '''"""Mock bias combination for testing."""
import jax.numpy as jnp

def BiasCombination(biases):
    """Mock PyBird bias combination function.

    Parameters
    ----------
    biases : array
        11 bias parameters: [b1, b2, b3, b4, b5, b6, b7, f, cϵ0, cϵ1, cϵ2]

    Returns
    -------
    array
        24 bias coefficients
    """
    b1, b2, b3, b4, b5, b6, b7, f, ce0, ce1, ce2 = biases
    return jnp.array([
        b1**2, 2*b1*f, f**2, 1., b1, b2, b3, b4,
        b1*b1, b1*b2, b1*b3, b1*b4,
        b2*b2, b2*b4, b4*b4,
        2*b1*b5, 2*b1*b6, 2*b1*b7,
        2*f*b5, 2*f*b6, 2*f*b7,
        ce0, ce1, ce2*f
    ])
'''


def create_mock_jacobian_bias_combination_file():
    """Create a mock Jacobian bias combination Python file content."""
    return '''"""Mock Jacobian bias combination for testing."""
import jax.numpy as jnp

def JacobianBiasCombination(biases):
    """Mock Jacobian of PyBird bias combination.

    Returns (24, 11) Jacobian matrix.
    """
    # Simplified mock - just return zeros for testing
    # Real implementation would have all derivatives
    return jnp.zeros((24, 11))
'''


def create_mock_stoch_model_file():
    """Create a mock StochModel Python file content."""
    return '''"""Mock StochModel for testing."""
import jax.numpy as jnp

def StochModel(k):
    """Mock stochastic model function.

    Parameters
    ----------
    k : array
        k-grid

    Returns
    -------
    array
        Stochastic components (len(k), 3)
    """
    n_k = len(k)
    km2 = 0.7**2
    k_rescaled = k * k / km2
    comp0 = jnp.ones(n_k)
    return jnp.column_stack((comp0, k_rescaled, k_rescaled/3))
'''


def create_mock_emulator_directory(base_path=None):
    """Create a complete mock emulator directory structure with all necessary files."""
    if base_path is None:
        base_path = tempfile.mkdtemp(prefix="mock_emulator_")
    else:
        os.makedirs(base_path, exist_ok=True)

    # Define proper output sizes for each component (matching PyBird structure)
    # For k-grid of 50 points: P11 has 4 components, Ploop has 22, Pct has 7
    component_specs = {
        "11": 4 * 50,     # P11: 4 components × 50 k-points = 200
        "loop": 22 * 50,  # Ploop: 22 components × 50 k-points = 1100
        "ct": 7 * 50      # Pct: 7 components × 50 k-points = 350
    }

    # Create component directories
    for component, n_outputs in component_specs.items():
        comp_path = Path(base_path) / component
        comp_path.mkdir(exist_ok=True)

        # Create component-specific nn_dict
        nn_dict = create_mock_nn_dict()
        nn_dict["n_output_features"] = n_outputs

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

        # Save postprocessing.py (needs to reshape output)
        with open(comp_path / "postprocessing.py", "w") as f:
            if component == "11":
                n_components = 4
            elif component == "loop":
                n_components = 22
            else:  # ct
                n_components = 7
            f.write(create_mock_postprocessing_file(n_components))

    # Save multipole-level bias combination (new PyBird-style)
    with open(Path(base_path) / "biascombination.py", "w") as f:
        f.write(create_mock_bias_combination_file())

    # Save Jacobian bias combination
    with open(Path(base_path) / "jacbiascombination.py", "w") as f:
        f.write(create_mock_jacobian_bias_combination_file())

    # Save StochModel
    with open(Path(base_path) / "stochmodel.py", "w") as f:
        f.write(create_mock_stoch_model_file())

    return base_path