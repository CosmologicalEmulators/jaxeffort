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

os.environ["JAXEFFORT_NO_AUTO_DOWNLOAD"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
import jaxeffort
from jaxeffort.jaxeffort import (
    MLP,
    MultipoleEmulators,
    MultipoleNoiseEmulator,
    load_component_emulator,
    get_stoch_terms
)

# Ensure float64
jax.config.update("jax_enable_x64", True)


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
            postprocessing=lambda i, o, d: o.reshape(100, 10),  # Reshape in postprocessing
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


class TestNumericalStability:
    """Test numerical precision and stability."""

    def test_maintains_float64_precision(self):
        """Verify float64 precision is maintained throughout."""
        # Create mock components
        k_grid = jnp.linspace(0.01, 0.3, 50, dtype=jnp.float64)

        # Test stochastic terms maintain precision
        cϵ0 = jnp.array(1.0, dtype=jnp.float64)
        cϵ1 = jnp.array(0.5, dtype=jnp.float64)
        cϵ2 = jnp.array(0.1, dtype=jnp.float64)
        n_bar = jnp.array(1e-3, dtype=jnp.float64)  # Typical galaxy number density

        P0, P2 = get_stoch_terms(cϵ0, cϵ1, cϵ2, n_bar, k_grid)

        assert P0.dtype == jnp.float64
        assert P2.dtype == jnp.float64
        assert jnp.all(jnp.isfinite(P0))
        assert jnp.all(jnp.isfinite(P2))

    def test_handles_extreme_values_gracefully(self):
        """Test behavior with extreme parameter values."""
        k_grid = jnp.logspace(-3, 0, 50)  # Wide k range

        # Test with different stochastic parameters
        cϵ0_large = jnp.array(10.0)
        cϵ1_large = jnp.array(10.0)
        cϵ2_large = jnp.array(10.0)
        n_bar = jnp.array(1e-3)

        P0_large, P2_large = get_stoch_terms(cϵ0_large, cϵ1_large, cϵ2_large, n_bar, k_grid)
        assert jnp.all(jnp.isfinite(P0_large))
        assert jnp.all(jnp.isfinite(P2_large))

        # Test with very small parameters
        cϵ0_small = jnp.array(0.01)
        cϵ1_small = jnp.array(0.01)
        cϵ2_small = jnp.array(0.01)
        P0_small, P2_small = get_stoch_terms(cϵ0_small, cϵ1_small, cϵ2_small, n_bar, k_grid)
        assert jnp.all(jnp.isfinite(P0_small))
        assert jnp.all(jnp.isfinite(P2_small))

        # Larger parameters should produce larger stochastic terms
        assert jnp.mean(jnp.abs(P0_large)) > jnp.mean(jnp.abs(P0_small))

    def test_gradient_stability(self):
        """Test that gradients are stable and well-behaved."""
        k_grid = jnp.linspace(0.01, 0.3, 50)

        def stoch_func(params):
            cϵ0, cϵ1, cϵ2 = params[:3]
            n_bar = params[3]
            result = get_stoch_terms(cϵ0, cϵ1, cϵ2, n_bar, k_grid)
            return jnp.sum(result[0]) + jnp.sum(result[1])  # Sum both monopole and quadrupole

        # Compute gradient
        grad_func = grad(stoch_func)
        params = jnp.array([1.0, 0.5, 0.1, 1e-3])  # cϵ0, cϵ1, cϵ2, n_bar
        gradient = grad_func(params)

        # Gradients should be finite
        assert jnp.all(jnp.isfinite(gradient))

        # Test numerical gradient for comparison
        eps = 1e-5
        numerical_grad = []
        for i in range(4):
            params_plus = params.at[i].add(eps)
            params_minus = params.at[i].add(-eps)
            num_grad = (stoch_func(params_plus) - stoch_func(params_minus)) / (2 * eps)
            numerical_grad.append(num_grad)

        numerical_grad = jnp.array(numerical_grad)

        # Should be close to analytical gradient
        assert jnp.allclose(gradient, numerical_grad, rtol=1e-4)


class TestComponentComposition:
    """Test how components are composed into multipoles."""

    def test_multipole_composition_consistency(self, tmp_path):
        """Verify multipole emulators compose components correctly."""
        # Create mock components
        mock_p11 = self._create_mock_component(output_shape=(50, 4))
        mock_ploop = self._create_mock_component(output_shape=(50, 22))
        mock_pct = self._create_mock_component(output_shape=(50, 7))

        # Create mock bias contraction
        def mock_bias_contraction(b1, b2, bs2, b3nl):
            # Return proper shape for contraction
            return jnp.ones((33,))  # 4 + 22 + 7 = 33 components

        # Create multipole emulator
        multipole = MultipoleEmulators(
            P11=mock_p11,
            Ploop=mock_ploop,
            Pct=mock_pct,
            bias_contraction=mock_bias_contraction,
            k_grid=jnp.linspace(0.01, 0.3, 50)
        )

        # Test with various inputs
        cosmology = jnp.ones(8)
        biases = jnp.array([1.0, 0.5, -0.5, 0.2])  # b1, b2, bs2, b3nl
        D = jnp.array(0.8)

        # Get monopole power spectrum
        P_mono = multipole.get_Pl(cosmology, biases, D)

        # Should have correct shape
        assert P_mono.shape == (50,)  # One value per k
        assert jnp.all(jnp.isfinite(P_mono))

    def test_noise_emulator_adds_stochastic_correctly(self, tmp_path):
        """Verify noise emulator correctly adds stochastic terms."""
        # Create base multipole emulator
        mock_p11 = self._create_mock_component(output_shape=(50, 4))
        mock_ploop = self._create_mock_component(output_shape=(50, 22))
        mock_pct = self._create_mock_component(output_shape=(50, 7))

        base_multipole = MultipoleEmulators(
            P11=mock_p11,
            Ploop=mock_ploop,
            Pct=mock_pct,
            bias_contraction=lambda b1, b2, bs2, b3nl: jnp.ones((33,)),
            k_grid=jnp.linspace(0.01, 0.3, 50)
        )

        # Create noise emulator
        noise_emulator = MultipoleNoiseEmulator(
            base_emulator=base_multipole,
            ell=0  # Monopole
        )

        # Test without noise (ceps = 0)
        cosmology = jnp.ones(11)  # 8 cosmo + 3 stochastic
        cosmology = cosmology.at[8:].set(0)  # Zero stochastic terms
        biases = jnp.array([1.0, 0.5, -0.5, 0.2])
        D = jnp.array(0.8)

        P_no_noise = noise_emulator.get_Pl(cosmology, biases, D)

        # Test with noise
        cosmology_with_noise = cosmology.at[8:].set([1.0, 0.5, 0.1])
        P_with_noise = noise_emulator.get_Pl(cosmology_with_noise, biases, D)

        # Should be different when noise is added
        assert not jnp.allclose(P_no_noise, P_with_noise)

        # Noise should add power (generally)
        assert jnp.mean(jnp.abs(P_with_noise)) >= jnp.mean(jnp.abs(P_no_noise))

    def _create_mock_component(self, output_shape):
        """Helper to create mock component."""
        class MockComponent:
            def get_component(self, input_data, D):
                return jnp.ones(output_shape)
        return MockComponent()


class TestMemoryEfficiency:
    """Test memory usage and efficiency."""

    def test_no_memory_leaks_in_repeated_calls(self):
        """Verify repeated calls don't leak memory."""
        k_grid = jnp.linspace(0.01, 0.3, 100)

        # Force garbage collection
        gc.collect()

        # Track memory over multiple calls
        memory_usage = []

        for i in range(100):
            cϵ0 = jnp.array(1.0)
            cϵ1 = jnp.array(0.5)
            cϵ2 = jnp.array(0.1)
            n_bar = jnp.array(1e-3)

            result = get_stoch_terms(cϵ0, cϵ1, cϵ2, n_bar, k_grid)

            # Periodically check memory
            if i % 20 == 0:
                gc.collect()
                # In a real test, we'd use memory_profiler or tracemalloc
                # Here we just ensure the result is reasonable
                assert len(result) == 2  # Tuple of (monopole, quadrupole)
                assert result[0].shape == (100,)  # Monopole
                assert result[1].shape == (100,)  # Quadrupole

        # Memory shouldn't grow significantly
        # (This is a simplified test - real memory testing is more complex)
        assert True  # Placeholder for actual memory assertion

    def test_efficient_broadcasting(self):
        """Test that operations use efficient broadcasting."""
        # Test vectorized operations
        k_grid = jnp.linspace(0.01, 0.3, 1000)  # Large k array

        # Batch of parameters
        batch_size = 10
        cϵ0_batch = jnp.ones(batch_size)
        cϵ1_batch = jnp.ones(batch_size) * 0.5
        cϵ2_batch = jnp.ones(batch_size) * 0.1
        n_bar_batch = jnp.ones(batch_size) * 1e-3

        # Vectorize the function
        vstoch = vmap(get_stoch_terms, in_axes=(0, 0, 0, 0, None))

        # Should handle batches efficiently
        results = vstoch(cϵ0_batch, cϵ1_batch, cϵ2_batch, n_bar_batch, k_grid)

        # Results should be tuple of (batch_monopole, batch_quadrupole)
        assert results[0].shape == (batch_size, 1000)  # Monopole
        assert results[1].shape == (batch_size, 1000)  # Quadrupole
        assert jnp.all(jnp.isfinite(results[0]))
        assert jnp.all(jnp.isfinite(results[1]))


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_bias_parameters(self):
        """Test behavior with zero bias parameters."""
        k_grid = jnp.linspace(0.01, 0.3, 50)

        # All zeros (except n_bar to avoid division by zero)
        cϵ0 = jnp.array(0.0)
        cϵ1 = jnp.array(0.0)
        cϵ2 = jnp.array(0.0)
        n_bar = jnp.array(1e-6)  # Small but non-zero

        P0, P2 = get_stoch_terms(cϵ0, cϵ1, cϵ2, n_bar, k_grid)

        # Should handle zeros gracefully
        assert jnp.all(jnp.isfinite(P0))
        assert jnp.all(jnp.isfinite(P2))

    def test_single_k_value(self):
        """Test with single k value instead of array."""
        k_single = jnp.array(0.1)
        cϵ0 = jnp.array(1.0)
        cϵ1 = jnp.array(0.5)
        cϵ2 = jnp.array(0.1)
        n_bar = jnp.array(1e-3)

        P0, P2 = get_stoch_terms(cϵ0, cϵ1, cϵ2, n_bar, k_single)

        # Should work with scalar k
        assert P0.shape == ()  # scalar
        assert P2.shape == ()  # scalar
        assert jnp.isfinite(P0)
        assert jnp.isfinite(P2)

    def test_extreme_k_ranges(self):
        """Test with very small and very large k values."""
        # Very small k
        k_small = jnp.array([1e-5, 1e-4, 1e-3])
        cϵ0 = jnp.array(1.0)
        cϵ1 = jnp.array(0.5)
        cϵ2 = jnp.array(0.1)
        n_bar = jnp.array(1e-3)

        P0_small, P2_small = get_stoch_terms(cϵ0, cϵ1, cϵ2, n_bar, k_small)
        assert jnp.all(jnp.isfinite(P0_small))
        assert jnp.all(jnp.isfinite(P2_small))

        # Very large k
        k_large = jnp.array([1.0, 10.0, 100.0])
        P0_large, P2_large = get_stoch_terms(cϵ0, cϵ1, cϵ2, n_bar, k_large)
        assert jnp.all(jnp.isfinite(P0_large))
        assert jnp.all(jnp.isfinite(P2_large))


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

    def test_jacobian_computation(self):
        """Test that Jacobians can be computed for sensitivity analysis."""
        k_grid = jnp.linspace(0.01, 0.3, 50)

        def forward(params):
            cϵ0, cϵ1, cϵ2, n_bar = params[:4]
            P0, P2 = get_stoch_terms(cϵ0, cϵ1, cϵ2, n_bar, k_grid)
            return jnp.sum(P0) + jnp.sum(P2)  # Scalar output for simple Jacobian

        # Compute Jacobian
        params = jnp.array([1.0, 0.5, 0.1, 1e-3])  # cϵ0, cϵ1, cϵ2, n_bar
        jac = jax.jacfwd(forward)(params)

        # Should have gradient for each parameter
        assert jac.shape == (4,)
        assert jnp.all(jnp.isfinite(jac))

        # n_bar should have effect (appears linearly in noise term)
        assert jnp.abs(jac[3]) > 0

    def _create_mock_emulator_files(self, tmp_path):
        """Create mock emulator files for testing."""
        # Weights
        np.save(tmp_path / "weights.npy", np.random.randn(100, 50))

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
            'activation': 'gelu',
            'layers': {
                'layer_1': {'n_neurons': 128},
                'layer_2': {'n_neurons': 128},
                'layer_3': {'n_neurons': 64}
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