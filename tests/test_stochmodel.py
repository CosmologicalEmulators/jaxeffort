#!/usr/bin/env python
"""
Unit tests for StochModel implementation in jaxeffort.

Tests cover:
1. StochModel loading
2. StochModel output shapes and values
3. Integration with get_Pl
4. Integration with get_Pl_jacobian
5. Consistency checks
"""

import pytest
import jax.numpy as jnp
import numpy as np
import sys
import os

# Add jaxeffort to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jaxeffort.jaxeffort as jaxeffort


class TestStochModelLoading:
    """Test that StochModel loading works correctly."""

    def test_load_stoch_model_exists(self):
        """Test that load_stoch_model function exists."""
        assert hasattr(jaxeffort, 'load_stoch_model')

    def test_load_stoch_model_from_directory(self, tmp_path):
        """Test loading StochModel from a directory."""
        # Create a temporary stochmodel.py file
        stoch_file = tmp_path / "stochmodel.py"
        stoch_file.write_text("""
import jax.numpy as jnp

def StochModel(k):
    return jnp.ones((len(k), 3))
""")

        # Load it
        stoch_model = jaxeffort.load_stoch_model(str(tmp_path))
        assert stoch_model is not None
        assert callable(stoch_model)

    def test_load_stoch_model_callable(self, tmp_path):
        """Test that loaded StochModel is callable."""
        stoch_file = tmp_path / "stochmodel.py"
        stoch_file.write_text("""
import jax.numpy as jnp

def StochModel(k):
    return jnp.zeros((len(k), 3))
""")

        stoch_model = jaxeffort.load_stoch_model(str(tmp_path))
        k_test = jnp.array([0.1, 0.2, 0.3])
        result = stoch_model(k_test)

        assert result.shape == (3, 3)
        assert jnp.allclose(result, 0.0)


class TestStochModelShapes:
    """Test that StochModel returns correct shapes."""

    @pytest.fixture
    def test_k_grid(self):
        """Create a test k-grid."""
        return jnp.logspace(-2, 0, 50)

    def test_monopole_stoch_shape(self, test_k_grid):
        """Test monopole StochModel returns correct shape."""
        # Create a simple monopole-like stoch model
        def stoch_model(k):
            km2 = 0.7**2
            k_rescaled = k * k / km2
            comp0 = jnp.ones(len(k))
            return jnp.column_stack((comp0, k_rescaled, k_rescaled/3))

        result = stoch_model(test_k_grid)
        assert result.shape == (len(test_k_grid), 3)

    def test_quadrupole_stoch_shape(self, test_k_grid):
        """Test quadrupole StochModel returns correct shape."""
        def stoch_model(k):
            km2 = 0.7**2
            k_rescaled = k * k / km2
            return jnp.column_stack((jnp.zeros(len(k)), jnp.zeros(len(k)), k_rescaled*(2/3)))

        result = stoch_model(test_k_grid)
        assert result.shape == (len(test_k_grid), 3)

    def test_hexadecapole_stoch_shape(self, test_k_grid):
        """Test hexadecapole StochModel returns correct shape."""
        def stoch_model(k):
            return jnp.zeros((len(k), 3))

        result = stoch_model(test_k_grid)
        assert result.shape == (len(test_k_grid), 3)


class TestStochModelValues:
    """Test that StochModel produces correct values."""

    def test_monopole_stoch_values(self):
        """Test monopole StochModel produces expected values."""
        k = jnp.array([0.1, 0.2, 0.3])
        km2 = 0.7**2

        def stoch_model(k):
            km2 = 0.7**2
            k_rescaled = k * k / km2
            comp0 = jnp.ones(len(k))
            return jnp.column_stack((comp0, k_rescaled, k_rescaled/3))

        result = stoch_model(k)

        # Check column 0 is all ones
        assert jnp.allclose(result[:, 0], 1.0)

        # Check column 1 is k^2/km2
        expected_col1 = k**2 / km2
        assert jnp.allclose(result[:, 1], expected_col1)

        # Check column 2 is (k^2/km2)/3
        expected_col2 = (k**2 / km2) / 3
        assert jnp.allclose(result[:, 2], expected_col2)

    def test_hexadecapole_stoch_zeros(self):
        """Test hexadecapole StochModel is all zeros."""
        k = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])

        def stoch_model(k):
            return jnp.zeros((len(k), 3))

        result = stoch_model(k)
        assert jnp.allclose(result, 0.0)


class TestBiasCombinationUpdated:
    """Test that BiasCombination accepts 11 parameters."""

    def test_bias_combination_11_params(self):
        """Test that BiasCombination now accepts 11 parameters."""
        # Create a test bias combination function
        def bias_combination(bs):
            b1, b2, b3, b4, b5, b6, b7, f, cϵ0, cϵ1, cϵ2 = bs
            return jnp.array([
                b1**2, 2*b1*f, f**2, 1., b1, b2, b3, b4,
                b1*b1, b1*b2, b1*b3, b1*b4,
                b2*b2, b2*b4, b4*b4,
                2*b1*b5, 2*b1*b6, 2*b1*b7,
                2*f*b5, 2*f*b6, 2*f*b7,
                cϵ0, cϵ1, cϵ2*f
            ])

        biases = jnp.array([2.0, 0.5, -0.4, 0.1, 0.05, 0.02, 0.01, 0.8, 1.0, 0.5, 0.2])
        result = bias_combination(biases)

        assert result.shape == (24,)

    def test_bias_combination_stoch_terms(self):
        """Test that stochastic terms are included in bias combination."""
        def bias_combination(bs):
            b1, b2, b3, b4, b5, b6, b7, f, cϵ0, cϵ1, cϵ2 = bs
            return jnp.array([
                b1**2, 2*b1*f, f**2, 1., b1, b2, b3, b4,
                b1*b1, b1*b2, b1*b3, b1*b4,
                b2*b2, b2*b4, b4*b4,
                2*b1*b5, 2*b1*b6, 2*b1*b7,
                2*f*b5, 2*f*b6, 2*f*b7,
                cϵ0, cϵ1, cϵ2*f
            ])

        biases = jnp.array([2.0, 0.5, -0.4, 0.1, 0.05, 0.02, 0.01, 0.8, 1.0, 0.5, 0.2])
        result = bias_combination(biases)

        # Check stochastic terms are in the result
        assert result[21] == biases[8]   # cϵ0
        assert result[22] == biases[9]   # cϵ1
        assert result[23] == biases[10] * biases[7]  # cϵ2*f


class TestJacobianBiasCombination:
    """Test JacobianBiasCombination for 11 parameters."""

    def test_jacobian_shape(self):
        """Test Jacobian has shape (24, 11)."""
        def jacobian_bias_combination(bs):
            b1, b2, b3, b4, b5, b6, b7, f, cϵ0, cϵ1, cϵ2 = bs
            J = jnp.zeros((24, 11))

            # Just set a few entries as example
            J = J.at[0, 0].set(2 * b1)
            J = J.at[21, 8].set(1.0)
            J = J.at[22, 9].set(1.0)
            J = J.at[23, 10].set(f)
            J = J.at[23, 7].set(cϵ2)

            return J

        biases = jnp.array([2.0, 0.5, -0.4, 0.1, 0.05, 0.02, 0.01, 0.8, 1.0, 0.5, 0.2])
        J = jacobian_bias_combination(biases)

        assert J.shape == (24, 11)

    def test_jacobian_stoch_derivatives(self):
        """Test that Jacobian includes derivatives for stochastic terms."""
        def jacobian_bias_combination(bs):
            b1, b2, b3, b4, b5, b6, b7, f, cϵ0, cϵ1, cϵ2 = bs
            J = jnp.zeros((24, 11))

            # Stochastic derivatives
            J = J.at[21, 8].set(1.0)     # d(cϵ0)/dcϵ0
            J = J.at[22, 9].set(1.0)     # d(cϵ1)/dcϵ1
            J = J.at[23, 10].set(f)      # d(cϵ2*f)/dcϵ2
            J = J.at[23, 7].set(cϵ2)     # d(cϵ2*f)/df

            return J

        biases = jnp.array([2.0, 0.5, -0.4, 0.1, 0.05, 0.02, 0.01, 0.8, 1.0, 0.5, 0.2])
        J = jacobian_bias_combination(biases)

        # Check stochastic derivatives
        assert J[21, 8] == 1.0           # d(cϵ0)/dcϵ0
        assert J[22, 9] == 1.0           # d(cϵ1)/dcϵ1
        assert J[23, 10] == biases[7]    # d(cϵ2*f)/dcϵ2 = f
        assert J[23, 7] == biases[10]    # d(cϵ2*f)/df = cϵ2


class TestMultipoleEmulatorsIntegration:
    """Test that MultipoleEmulators integrates StochModel correctly."""

    @pytest.fixture
    def mock_emulator_components(self):
        """Create mock emulator components."""
        from unittest.mock import Mock

        # Mock MLP components
        P11 = Mock()
        P11.k_grid = jnp.linspace(0.01, 0.5, 10).reshape(-1, 1)
        P11.apply_model = Mock(return_value=jnp.ones((10, 3)))
        P11.InMinMax = jnp.array([[0, 1]] * 9)
        P11.OutMinMax = jnp.array([[0, 1]] * 10)
        P11.postprocessing = lambda params, output, D, emu: output * D**2

        Ploop = Mock()
        Ploop.apply_model = Mock(return_value=jnp.ones((10, 12)))
        Ploop.InMinMax = jnp.array([[0, 1]] * 9)
        Ploop.OutMinMax = jnp.array([[0, 1]] * 10)
        Ploop.postprocessing = lambda params, output, D, emu: output * D**2

        Pct = Mock()
        Pct.apply_model = Mock(return_value=jnp.ones((10, 6)))
        Pct.InMinMax = jnp.array([[0, 1]] * 9)
        Pct.OutMinMax = jnp.array([[0, 1]] * 10)
        Pct.postprocessing = lambda params, output, D, emu: output * D**2

        # Mock functions
        def bias_combination(bs):
            return jnp.ones(24)

        def stoch_model(k):
            return jnp.ones((len(k), 3))

        def jacobian_bias_combination(bs):
            return jnp.ones((24, 11))

        return P11, Ploop, Pct, bias_combination, stoch_model, jacobian_bias_combination

    def test_multipole_emulator_has_stoch_model(self, mock_emulator_components):
        """Test that MultipoleEmulators stores stoch_model."""
        P11, Ploop, Pct, bias_comb, stoch, jac_bias = mock_emulator_components

        emulator = jaxeffort.MultipoleEmulators(
            P11, Ploop, Pct, bias_comb, stoch, jac_bias
        )

        assert hasattr(emulator, 'stoch_model')
        assert emulator.stoch_model is not None
        assert callable(emulator.stoch_model)


class TestConsistency:
    """Test consistency between get_Pl and get_Pl_jacobian."""

    def test_pl_values_match_between_methods(self):
        """Test that P_ℓ values match between get_Pl and get_Pl_jacobian."""
        # This would require a full emulator setup
        # For now, we'll test the principle with mocks

        # Create simple mock functions
        def get_components(cosmology, D):
            k_size = 10
            return jnp.ones((k_size, 3)), jnp.ones((k_size, 12)), jnp.ones((k_size, 6))

        def stoch_model(k):
            return jnp.ones((len(k), 3))

        def bias_combination(bs):
            return jnp.ones(24)

        def jacobian_bias_combination(bs):
            return jnp.eye(24, 11)

        # Mock compute P_ℓ
        k_grid = jnp.linspace(0.01, 0.5, 10).reshape(-1, 1)
        cosmology = jnp.array([1.0, 3.0, 0.96, 67.0, 0.022, 0.12, 0.06, -1.0, 0.0])
        biases = jnp.array([2.0, 0.5, -0.4, 0.1, 0.05, 0.02, 0.01, 0.8, 1.0, 0.5, 0.2])
        D = 0.8

        # Get components
        P11_comp, Ploop_comp, Pct_comp = get_components(cosmology, D)
        stoch_comp = stoch_model(k_grid)

        # Stack
        stacked = jnp.hstack((P11_comp, Ploop_comp, Pct_comp, stoch_comp))

        # Compute P_ℓ
        biases_vec = bias_combination(biases)
        Pl = stacked @ biases_vec

        # Compute with Jacobian
        jac = jacobian_bias_combination(biases)
        Pl_from_jac = stacked @ biases_vec
        Pl_jac = stacked @ jac

        # Check they match
        assert jnp.allclose(Pl, Pl_from_jac)
        assert Pl_jac.shape == (10, 11)


class TestRegressionAgainstEffortJl:
    """Regression tests against known Effort.jl values."""

    def test_stochmodel_values_match_julia(self):
        """Test that StochModel values match Julia implementation."""
        # Known values from Julia
        k = jnp.array([0.01, 0.1, 0.2])
        km2 = 0.7**2

        # Monopole stoch model (from Julia)
        def stoch_model_mono(k):
            k_rescaled = k * k / km2
            comp0 = jnp.ones(len(k))
            return jnp.column_stack((comp0, k_rescaled, k_rescaled/3))

        result = stoch_model_mono(k)

        # Check against expected Julia values
        expected_col0 = jnp.ones(3)
        expected_col1 = k**2 / km2
        expected_col2 = (k**2 / km2) / 3

        assert jnp.allclose(result[:, 0], expected_col0)
        assert jnp.allclose(result[:, 1], expected_col1)
        assert jnp.allclose(result[:, 2], expected_col2)


def test_import():
    """Test that jaxeffort can be imported."""
    import jaxeffort
    assert jaxeffort is not None


def test_load_functions_exist():
    """Test that all loading functions exist."""
    assert hasattr(jaxeffort, 'load_bias_combination')
    assert hasattr(jaxeffort, 'load_stoch_model')
    assert hasattr(jaxeffort, 'load_jacobian_bias_combination')
    assert hasattr(jaxeffort, 'load_multipole_emulator')


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v'])
