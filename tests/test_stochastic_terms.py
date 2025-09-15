"""
Test stochastic term functions for EFT power spectrum calculations.
"""

import pytest
import numpy as np
import jax.numpy as jnp
import jax

from jaxeffort import get_stoch_terms

# Configure JAX for 64-bit precision
jax.config.update('jax_enable_x64', True)


class TestStochasticTerms:
    """Test basic stochastic term calculations."""

    def test_get_stoch_terms_basic(self):
        """Test basic stochastic terms computation."""
        # Set up test parameters
        cϵ0 = 1.0
        cϵ1 = 2.0
        cϵ2 = 0.5
        n_bar = 1e-3  # galaxies per (Mpc/h)^3
        k_grid = jnp.array([0.01, 0.05, 0.1, 0.2, 0.3])
        k_nl = 0.7

        # Compute stochastic terms
        P_stoch_0, P_stoch_2 = get_stoch_terms(cϵ0, cϵ1, cϵ2, n_bar, k_grid, k_nl)

        # Check shapes
        assert P_stoch_0.shape == k_grid.shape
        assert P_stoch_2.shape == k_grid.shape

        # Check that values are positive (for positive parameters)
        assert np.all(P_stoch_0 > 0)
        assert np.all(P_stoch_2 >= 0)

        # Check specific values
        # P_stoch_0 = (1/n_bar) * (cϵ0 + cϵ1 * (k/k_nl)^2)
        expected_P0_first = (1 / n_bar) * (cϵ0 + cϵ1 * (k_grid[0] / k_nl)**2)
        assert np.isclose(P_stoch_0[0], expected_P0_first)

        # P_stoch_2 = (1/n_bar) * (cϵ2 * (k/k_nl)^2)
        expected_P2_first = (1 / n_bar) * (cϵ2 * (k_grid[0] / k_nl)**2)
        assert np.isclose(P_stoch_2[0], expected_P2_first)

    def test_get_stoch_terms_zero_parameters(self):
        """Test stochastic terms with zero parameters."""
        n_bar = 1e-3
        k_grid = jnp.array([0.01, 0.1, 0.3])

        # Test with cϵ0 = 0
        P_stoch_0, P_stoch_2 = get_stoch_terms(0.0, 1.0, 1.0, n_bar, k_grid)
        assert P_stoch_0[0] < P_stoch_0[-1]  # Should increase with k

        # Test with cϵ1 = 0
        P_stoch_0, P_stoch_2 = get_stoch_terms(1.0, 0.0, 1.0, n_bar, k_grid)
        assert np.allclose(P_stoch_0, 1.0 / n_bar)  # Should be constant

        # Test with cϵ2 = 0
        P_stoch_0, P_stoch_2 = get_stoch_terms(1.0, 1.0, 0.0, n_bar, k_grid)
        assert np.allclose(P_stoch_2, 0.0)  # Should be zero

    def test_get_stoch_terms_scaling(self):
        """Test scaling behavior of stochastic terms."""
        cϵ0 = 1.0
        cϵ1 = 2.0
        cϵ2 = 0.5
        n_bar = 1e-3
        k_grid = jnp.array([0.01, 0.1, 0.3])
        k_nl = 0.7

        # Compute with default parameters
        P0_ref, P2_ref = get_stoch_terms(cϵ0, cϵ1, cϵ2, n_bar, k_grid, k_nl)

        # Double n_bar should halve the power
        P0_double_n, P2_double_n = get_stoch_terms(cϵ0, cϵ1, cϵ2, 2 * n_bar, k_grid, k_nl)
        assert np.allclose(P0_double_n, P0_ref / 2)
        assert np.allclose(P2_double_n, P2_ref / 2)

        # Double k_nl should reduce k^2 dependence
        P0_double_knl, P2_double_knl = get_stoch_terms(cϵ0, cϵ1, cϵ2, n_bar, k_grid, 2 * k_nl)
        # The k^2 term should be reduced by a factor of 4
        assert np.all(P0_double_knl < P0_ref)
        assert np.all(P2_double_knl < P2_ref)


class TestStochasticTermsEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_k_grid(self):
        """Test with empty k grid."""
        cϵ0 = 1.0
        cϵ1 = 2.0
        cϵ2 = 0.5
        n_bar = 1e-3
        k_grid = jnp.array([])

        P_stoch_0, P_stoch_2 = get_stoch_terms(cϵ0, cϵ1, cϵ2, n_bar, k_grid)

        assert P_stoch_0.shape == (0,)
        assert P_stoch_2.shape == (0,)

    def test_single_k_value(self):
        """Test with single k value."""
        cϵ0 = 1.0
        cϵ1 = 2.0
        cϵ2 = 0.5
        n_bar = 1e-3
        k_grid = jnp.array([0.1])

        P_stoch_0, P_stoch_2 = get_stoch_terms(cϵ0, cϵ1, cϵ2, n_bar, k_grid)

        assert P_stoch_0.shape == (1,)
        assert P_stoch_2.shape == (1,)
        assert P_stoch_0[0] > 0
        assert P_stoch_2[0] >= 0

    def test_high_k_behavior(self):
        """Test behavior at high k values."""
        cϵ0 = 1.0
        cϵ1 = 2.0
        cϵ2 = 0.5
        n_bar = 1e-3
        k_grid = jnp.logspace(-2, 1, 10)  # 0.01 to 10

        P_stoch_0, P_stoch_2 = get_stoch_terms(cϵ0, cϵ1, cϵ2, n_bar, k_grid)

        # At high k, k^2 term should dominate
        k2_ratio = (k_grid[-1] / k_grid[0])**2
        P0_ratio = P_stoch_0[-1] / P_stoch_0[0]
        P2_ratio = P_stoch_2[-1] / P_stoch_2[0]

        # The ratio should be close to k^2 ratio for large k
        # (when k^2 term dominates over constant term)
        assert P2_ratio > 0.5 * k2_ratio  # P2 has only k^2 term

    def test_vectorized_inputs(self):
        """Test with vectorized inputs."""
        cϵ0 = 1.0
        cϵ1 = 2.0
        cϵ2 = 0.5
        n_bar = 1e-3
        k_grid = jnp.linspace(0.01, 0.5, 100)

        P_stoch_0, P_stoch_2 = get_stoch_terms(cϵ0, cϵ1, cϵ2, n_bar, k_grid)

        # Should work with large arrays
        assert P_stoch_0.shape == (100,)
        assert P_stoch_2.shape == (100,)

        # Should be monotonically increasing (for positive cϵ1, cϵ2)
        assert np.all(np.diff(P_stoch_0) > 0)
        assert np.all(np.diff(P_stoch_2) > 0)


class TestJAXFeatures:
    """Test JAX-specific features like JIT compilation and gradients."""

    def test_jit_compilation(self):
        """Test that functions work with JIT compilation."""
        import jax

        # JIT compile the function
        get_stoch_terms_jit = jax.jit(get_stoch_terms)

        cϵ0 = 1.0
        cϵ1 = 2.0
        cϵ2 = 0.5
        n_bar = 1e-3
        k_grid = jnp.array([0.01, 0.1, 0.3])

        # Should produce same results
        P0_normal, P2_normal = get_stoch_terms(cϵ0, cϵ1, cϵ2, n_bar, k_grid)
        P0_jit, P2_jit = get_stoch_terms_jit(cϵ0, cϵ1, cϵ2, n_bar, k_grid)

        assert np.allclose(P0_normal, P0_jit)
        assert np.allclose(P2_normal, P2_jit)

    def test_gradient_computation(self):
        """Test that gradients can be computed."""
        import jax

        def loss_fn(params):
            cϵ0, cϵ1, cϵ2 = params
            n_bar = 1e-3
            k_grid = jnp.array([0.1])
            P0, P2 = get_stoch_terms(cϵ0, cϵ1, cϵ2, n_bar, k_grid)
            return P0[0] + P2[0]

        params = jnp.array([1.0, 2.0, 0.5])
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params)

        # Should have gradients for all parameters
        assert grads.shape == params.shape
        assert not np.any(np.isnan(grads))

    def test_vmap_batch_processing(self):
        """Test batch processing with vmap."""
        import jax

        # Create batch of parameters
        cϵ0_batch = jnp.array([1.0, 2.0, 3.0])
        cϵ1_batch = jnp.array([2.0, 3.0, 4.0])
        cϵ2_batch = jnp.array([0.5, 1.0, 1.5])
        n_bar = 1e-3
        k_grid = jnp.array([0.01, 0.1, 0.3])

        # Define function for single set of parameters
        def single_stoch(cϵ0, cϵ1, cϵ2):
            return get_stoch_terms(cϵ0, cϵ1, cϵ2, n_bar, k_grid)

        # Vectorize over batch dimension
        batch_stoch = jax.vmap(single_stoch, in_axes=(0, 0, 0))
        P0_batch, P2_batch = batch_stoch(cϵ0_batch, cϵ1_batch, cϵ2_batch)

        # Check shapes
        assert P0_batch.shape == (3, 3)  # 3 parameter sets, 3 k values
        assert P2_batch.shape == (3, 3)


class TestComparisonWithEffortJl:
    """Test that results match expectations from Effort.jl."""

    def test_monopole_quadrupole_relationship(self):
        """Test relationship between monopole and quadrupole terms."""
        cϵ0 = 1.0
        cϵ1 = 0.0  # No k^2 term in monopole
        cϵ2 = 2.0
        n_bar = 1e-3
        k_grid = jnp.array([0.01, 0.1, 0.3])

        P_stoch_0, P_stoch_2 = get_stoch_terms(cϵ0, cϵ1, cϵ2, n_bar, k_grid)

        # Monopole should be constant (only cϵ0 term)
        assert np.allclose(P_stoch_0, cϵ0 / n_bar)

        # Quadrupole should scale as k^2
        k_ratio = k_grid[-1] / k_grid[0]
        P2_ratio = P_stoch_2[-1] / P_stoch_2[0]
        assert np.isclose(P2_ratio, k_ratio**2, rtol=1e-6)

    def test_numerical_values(self):
        """Test specific numerical values for validation."""
        cϵ0 = 100.0
        cϵ1 = 200.0
        cϵ2 = 50.0
        n_bar = 2.5e-4  # Typical galaxy number density
        k_nl = 0.7
        k = 0.1  # Test at specific k

        P_stoch_0, P_stoch_2 = get_stoch_terms(cϵ0, cϵ1, cϵ2, n_bar, jnp.array([k]), k_nl)

        # Manual calculation
        expected_P0 = (1 / n_bar) * (cϵ0 + cϵ1 * (k / k_nl)**2)
        expected_P2 = (1 / n_bar) * (cϵ2 * (k / k_nl)**2)

        assert np.isclose(P_stoch_0[0], expected_P0, rtol=1e-10)
        assert np.isclose(P_stoch_2[0], expected_P2, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])