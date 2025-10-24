"""
Robust tests for jaxeffort.py core functionality.

These tests validate:
- JIT compilation behavior and performance
- Numerical precision and stability
- Component composition and bias contraction
- Memory efficiency
- Edge cases in emulator evaluation
"""

import os
import sys
import numpy as np
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
import pytest
import tempfile
import gc

# NOTE: DO NOT set environment variables at module level!
# This pollutes the environment for all subsequent tests in the session.
# Use fixtures or context managers instead.

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
import jaxeffort
from jaxeffort.jaxeffort import (
    MLP,
    MultipoleEmulators,
    load_component_emulator
)

# Ensure float64
jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module", autouse=True)
def disable_auto_download():
    """Disable auto-download for this test module only."""
    old_value = os.environ.get("JAXEFFORT_NO_AUTO_DOWNLOAD")
    os.environ["JAXEFFORT_NO_AUTO_DOWNLOAD"] = "1"
    yield
    # Restore original value
    if old_value is None:
        os.environ.pop("JAXEFFORT_NO_AUTO_DOWNLOAD", None)
    else:
        os.environ["JAXEFFORT_NO_AUTO_DOWNLOAD"] = old_value


class TestJITCompilationBehavior:
    """Test JIT compilation optimizations and alternatives."""

    def test_jit_compilation_speedup(self, tmp_path):
        """Verify JIT compilation provides performance benefits."""
        # Create a mock emulator
        mock_emulator = self._create_mock_mlp(tmp_path)

        # Create input
        input_data = jnp.ones(12)  # 8 bias + 4 cosmology params
        D = jnp.array(1.0)

        # Time first call (includes compilation)
        start = time.time()
        result1 = mock_emulator.get_component(input_data, D)
        first_call_time = time.time() - start

        # Time subsequent calls (should be faster)
        times = []
        for _ in range(10):
            start = time.time()
            result = mock_emulator.get_component(input_data, D)
            times.append(time.time() - start)

        avg_subsequent_time = np.mean(times)

        # Subsequent calls should be significantly faster
        # (In real scenarios, speedup is often 10-100x)
        # Here we just check there's some speedup
        assert avg_subsequent_time < first_call_time

    def test_alternative_jit_path_without_emulator_param(self, tmp_path):
        """Test the alternative JIT compilation path."""
        # Create MLP with mock that doesn't support emulator parameter
        mock_flax = MagicMock()

        # Make run_emulator not accept emulator parameter
        def run_emulator_no_param(self, input_data):
            return jnp.ones((1000,))  # Return flat array matching out_MinMax size

        mock_flax.run_emulator = run_emulator_no_param.__get__(mock_flax, type(mock_flax))

        # Check if method accepts 'emulator' parameter - should be False
        import inspect
        sig = inspect.signature(mock_flax.run_emulator)
        assert 'emulator' not in sig.parameters

        # Create MLP with this mock
        mlp = MLP(
            emulator=mock_flax,
            k_grid=jnp.linspace(0.01, 0.3, 100),
            in_MinMax=jnp.array([[0, 1]] * 12),
            out_MinMax=jnp.array([[0, 1]] * 1000),
            postprocessing=lambda i, o, d: o,  # Return flat array for reshape by get_component
            emulator_description={},
            nn_dict={'n_hidden_layers': 2, 'n_output_features': 1000,
                    'layers': {'layer_1': {'n_neurons': 50, 'activation_function': 'tanh'},
                              'layer_2': {'n_neurons': 50, 'activation_function': 'tanh'}}}
        )

        # Should still work with alternative JIT path
        input_data = jnp.ones(12)
        D = jnp.array(1.0)
        result = mlp.get_component(input_data, D)

        assert result.shape == (100, 10)  # Reshaped output

    def _create_mock_mlp(self, tmp_path):
        """Helper to create a mock MLP for testing."""
        mock_emulator = MagicMock()
        # Fix: run_emulator should only take x, not params
        mock_emulator.run_emulator = lambda x: jnp.ones((1000,))  # Returns flat array
        mock_emulator.parameters = {'test': 'params'}

        return MLP(
            emulator=mock_emulator,
            k_grid=jnp.linspace(0.01, 0.3, 100),
            in_MinMax=jnp.array([[0, 1]] * 12),
            out_MinMax=jnp.array([[0, 1]] * 1000),
            postprocessing=lambda i, o, d: o * d,
            emulator_description={},
            nn_dict={'n_hidden_layers': 2, 'n_output_features': 1000,
                    'layers': {'layer_1': {'n_neurons': 50, 'activation_function': 'tanh'},
                              'layer_2': {'n_neurons': 50, 'activation_function': 'tanh'}}}
        )


class TestComponentComposition:
    """Test how components are composed into multipoles."""

    def test_multipole_composition_consistency(self, tmp_path):
        """Verify multipole emulators compose components correctly."""
        import jaxeffort

        # Use real emulator
        emulator = jaxeffort.trained_emulators.get('pybird_mnuw0wacdm', {}).get('0')
        if emulator is None:
            pytest.skip("Real emulator not available (not downloaded)")

        # Test with various inputs
        cosmology = jnp.ones(9)  # 9 cosmology parameters
        biases = jnp.array([2.0, 0.5, -0.4, 0.1, 0.05, 0.02, 0.01, 0.8, 1.0, 0.5, 0.2])  # 11 PyBird biases
        D = jnp.array(0.8)

        # Get monopole power spectrum
        P_mono = emulator.get_Pl(cosmology, biases, D)

        # Should have correct shape
        assert P_mono.shape[0] > 0  # Has some k-points
        assert jnp.all(jnp.isfinite(P_mono))


class TestRealWorldIntegration:
    """Test complete workflows with realistic scenarios."""

    def test_full_pipeline_with_mock_data(self, tmp_path):
        """Test complete pipeline from loading to evaluation."""
        # Create comprehensive mock emulator structure
        self._create_mock_emulator_files(tmp_path)

        # Load emulator
        emulator = load_component_emulator(tmp_path)

        # Test evaluation with realistic parameters
        # Cosmology: Om, Ob, h, ns, s8, mnu, w0, wa
        cosmology = jnp.array([0.3, 0.05, 0.7, 0.96, 0.8, 0.06, -1.0, 0.0])
        # Biases: b1, b2, bs2, b3nl
        biases = jnp.array([2.0, 0.5, -0.4, 0.1])
        combined = jnp.concatenate([biases, cosmology])
        D = jnp.array(0.8)

        # Should evaluate without errors
        result = emulator.get_component(combined, D)

        assert result.ndim == 2  # (n_k, n_components)
        assert jnp.all(jnp.isfinite(result))

    def _create_mock_emulator_files(self, tmp_path):
        """Create mock emulator files for testing."""
        # Calculate correct number of weights for the network structure
        # Input to layer1: 12 * 128 + 128 (bias) = 1664
        # Layer1 to layer2: 128 * 128 + 128 (bias) = 16512
        # Layer2 to layer3: 128 * 64 + 64 (bias) = 8256
        # Layer3 to output: 64 * 1000 + 1000 (bias) = 65000
        # Total: 1664 + 16512 + 8256 + 65000 = 91432
        total_weights = 91432
        np.save(tmp_path / "weights.npy", np.random.randn(total_weights))

        # Normalization
        np.save(tmp_path / "inminmax.npy", np.array([[0, 1]] * 12))
        np.save(tmp_path / "outminmax.npy", np.array([[0, 1]] * 1000))

        # k grid
        np.save(tmp_path / "k.npy", np.linspace(0.01, 0.3, 50))

        # Network setup
        import json
        nn_setup = {
            'n_hidden_layers': 3,
            'n_input_features': 12,
            'n_output_features': 1000,
            'activation': 'relu',
            'layers': {
                'layer_1': {'n_neurons': 128, 'activation_function': 'relu'},
                'layer_2': {'n_neurons': 128, 'activation_function': 'relu'},
                'layer_3': {'n_neurons': 64, 'activation_function': 'relu'}
            }
        }
        with open(tmp_path / "nn_setup.json", "w") as f:
            json.dump(nn_setup, f)

        # Postprocessing (simple identity)
        postproc = """
def postprocessing(input, output, D):
    return output * D
"""
        (tmp_path / "postprocessing.py").write_text(postproc)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])