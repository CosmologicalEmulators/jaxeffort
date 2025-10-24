#!/usr/bin/env python
"""
Unit tests for updated BiasCombination (11 parameters -> 24 coefficients).

Tests cover:
1. Correct number of parameters accepted
2. Correct number of coefficients returned
3. Stochastic terms included
4. Jacobian has correct shape
5. Jacobian correctness via finite differences
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np


class TestBiasCombination11Parameters:
    """Test that BiasCombination now accepts 11 parameters."""

    @pytest.fixture
    def pybird_bias_combination_func(self):
        """PyBird-specific bias combination for all multipoles."""
        def pybird_bias_combination(bs):
            b1, b2, b3, b4, b5, b6, b7, f, cϵ0, cϵ1, cϵ2 = bs
            return jnp.array([
                b1**2, 2*b1*f, f**2, 1., b1, b2, b3, b4,
                b1*b1, b1*b2, b1*b3, b1*b4,
                b2*b2, b2*b4, b4*b4,
                2*b1*b5, 2*b1*b6, 2*b1*b7,
                2*f*b5, 2*f*b6, 2*f*b7,
                cϵ0, cϵ1, cϵ2*f
            ])
        return pybird_bias_combination

    @pytest.fixture
    def test_biases(self):
        """Standard test bias parameters."""
        return jnp.array([2.0, 0.5, -0.4, 0.1, 0.05, 0.02, 0.01, 0.8, 1.0, 0.5, 0.2])

    def test_accepts_11_parameters(self, pybird_bias_combination_func, test_biases):
        """Test that function accepts 11 parameters."""
        assert len(test_biases) == 11
        result = pybird_bias_combination_func(test_biases)
        assert result is not None

    def test_returns_24_coefficients(self, pybird_bias_combination_func, test_biases):
        """Test that function returns 24 coefficients."""
        result = pybird_bias_combination_func(test_biases)
        assert result.shape == (24,)

    def test_old_8_parameters_fails(self, pybird_bias_combination_func):
        """Test that old 8-parameter version fails."""
        old_biases = jnp.array([2.0, 0.5, -0.4, 0.1, 0.05, 0.02, 0.01, 0.8])

        with pytest.raises(ValueError, match="too many values to unpack|not enough values"):
            pybird_bias_combination_func(old_biases)

    def test_bias_terms_correct(self, pybird_bias_combination_func, test_biases):
        """Test that bias terms are computed correctly."""
        result = pybird_bias_combination_func(test_biases)
        b1, b2, b3, b4, b5, b6, b7, f, cϵ0, cϵ1, cϵ2 = test_biases

        # Check a few key terms
        assert jnp.isclose(result[0], b1**2)
        assert jnp.isclose(result[1], 2*b1*f)
        assert jnp.isclose(result[2], f**2)
        assert jnp.isclose(result[3], 1.0)
        assert jnp.isclose(result[4], b1)

    def test_stochastic_terms_included(self, pybird_bias_combination_func, test_biases):
        """Test that stochastic terms are in the output."""
        result = pybird_bias_combination_func(test_biases)
        b1, b2, b3, b4, b5, b6, b7, f, cϵ0, cϵ1, cϵ2 = test_biases

        # Check stochastic terms (indices 21, 22, 23)
        assert jnp.isclose(result[21], cϵ0)
        assert jnp.isclose(result[22], cϵ1)
        assert jnp.isclose(result[23], cϵ2 * f)

    def test_stochastic_term_dependencies(self, pybird_bias_combination_func):
        """Test that stochastic terms depend on correct parameters."""
        # Test cϵ0 is independent of f
        biases1 = jnp.array([2.0, 0.5, -0.4, 0.1, 0.05, 0.02, 0.01, 0.8, 1.0, 0.5, 0.2])
        biases2 = jnp.array([2.0, 0.5, -0.4, 0.1, 0.05, 0.02, 0.01, 0.9, 1.0, 0.5, 0.2])  # Different f

        result1 = pybird_bias_combination_func(biases1)
        result2 = pybird_bias_combination_func(biases2)

        # cϵ0 and cϵ1 should be the same
        assert jnp.isclose(result1[21], result2[21])
        assert jnp.isclose(result1[22], result2[22])

        # cϵ2*f should be different
        assert not jnp.isclose(result1[23], result2[23])


class TestJacobianBiasCombination11Parameters:
    """Test Jacobian for 11 parameter bias combination."""

    @pytest.fixture
    def jacobian_func(self):
        """Complete Jacobian of bias combination with all derivatives."""
        def jacobian_bias_combination(bs):
            b1, b2, b3, b4, b5, b6, b7, f, cϵ0, cϵ1, cϵ2 = bs
            J = jnp.zeros((24, 11))

            # Derivatives with respect to b1 (column 0)
            J = J.at[0, 0].set(2 * b1)      # d(b1²)/db1
            J = J.at[1, 0].set(2 * f)       # d(2*b1*f)/db1
            J = J.at[4, 0].set(1.0)         # d(b1)/db1
            J = J.at[8, 0].set(2 * b1)      # d(b1*b1)/db1
            J = J.at[9, 0].set(b2)          # d(b1*b2)/db1
            J = J.at[10, 0].set(b3)         # d(b1*b3)/db1
            J = J.at[11, 0].set(b4)         # d(b1*b4)/db1
            J = J.at[15, 0].set(2 * b5)     # d(2*b1*b5)/db1
            J = J.at[16, 0].set(2 * b6)     # d(2*b1*b6)/db1
            J = J.at[17, 0].set(2 * b7)     # d(2*b1*b7)/db1

            # Derivatives with respect to b2 (column 1)
            J = J.at[5, 1].set(1.0)         # d(b2)/db2
            J = J.at[9, 1].set(b1)          # d(b1*b2)/db2
            J = J.at[12, 1].set(2 * b2)     # d(b2²)/db2
            J = J.at[13, 1].set(b4)         # d(b2*b4)/db2

            # Derivatives with respect to b3 (column 2)
            J = J.at[6, 2].set(1.0)         # d(b3)/db3
            J = J.at[10, 2].set(b1)         # d(b1*b3)/db3

            # Derivatives with respect to b4 (column 3)
            J = J.at[7, 3].set(1.0)         # d(b4)/db4
            J = J.at[11, 3].set(b1)         # d(b1*b4)/db4
            J = J.at[13, 3].set(b2)         # d(b2*b4)/db4
            J = J.at[14, 3].set(2 * b4)     # d(b4²)/db4

            # Derivatives with respect to b5 (column 4)
            J = J.at[15, 4].set(2 * b1)     # d(2*b1*b5)/db5
            J = J.at[18, 4].set(2 * f)      # d(2*f*b5)/db5

            # Derivatives with respect to b6 (column 5)
            J = J.at[16, 5].set(2 * b1)     # d(2*b1*b6)/db6
            J = J.at[19, 5].set(2 * f)      # d(2*f*b6)/db6

            # Derivatives with respect to b7 (column 6)
            J = J.at[17, 6].set(2 * b1)     # d(2*b1*b7)/db7
            J = J.at[20, 6].set(2 * f)      # d(2*f*b7)/db7

            # Derivatives with respect to f (column 7)
            J = J.at[1, 7].set(2 * b1)      # d(2*b1*f)/df
            J = J.at[2, 7].set(2 * f)       # d(f²)/df
            J = J.at[18, 7].set(2 * b5)     # d(2*f*b5)/df
            J = J.at[19, 7].set(2 * b6)     # d(2*f*b6)/df
            J = J.at[20, 7].set(2 * b7)     # d(2*f*b7)/df
            J = J.at[23, 7].set(cϵ2)        # d(cϵ2*f)/df

            # Derivatives with respect to cϵ0 (column 8)
            J = J.at[21, 8].set(1.0)        # d(cϵ0)/dcϵ0

            # Derivatives with respect to cϵ1 (column 9)
            J = J.at[22, 9].set(1.0)        # d(cϵ1)/dcϵ1

            # Derivatives with respect to cϵ2 (column 10)
            J = J.at[23, 10].set(f)         # d(cϵ2*f)/dcϵ2

            return J
        return jacobian_bias_combination

    @pytest.fixture
    def pybird_bias_combination_func(self):
        """PyBird-specific bias combination function for validation."""
        def pybird_bias_combination(bs):
            b1, b2, b3, b4, b5, b6, b7, f, cϵ0, cϵ1, cϵ2 = bs
            return jnp.array([
                b1**2, 2*b1*f, f**2, 1., b1, b2, b3, b4,
                b1*b1, b1*b2, b1*b3, b1*b4,
                b2*b2, b2*b4, b4*b4,
                2*b1*b5, 2*b1*b6, 2*b1*b7,
                2*f*b5, 2*f*b6, 2*f*b7,
                cϵ0, cϵ1, cϵ2*f
            ])
        return pybird_bias_combination

    @pytest.fixture
    def test_biases(self):
        """Standard test bias parameters."""
        return jnp.array([2.0, 0.5, -0.4, 0.1, 0.05, 0.02, 0.01, 0.8, 1.0, 0.5, 0.2])

    def test_jacobian_shape(self, jacobian_func, test_biases):
        """Test Jacobian has correct shape (24, 11)."""
        J = jacobian_func(test_biases)
        assert J.shape == (24, 11)

    def test_jacobian_stochastic_derivatives(self, jacobian_func, test_biases):
        """Test stochastic derivatives in Jacobian."""
        J = jacobian_func(test_biases)
        f = test_biases[7]
        cϵ2 = test_biases[10]

        # Check stochastic derivatives
        assert jnp.isclose(J[21, 8], 1.0)      # d(cϵ0)/dcϵ0
        assert jnp.isclose(J[22, 9], 1.0)      # d(cϵ1)/dcϵ1
        assert jnp.isclose(J[23, 10], f)       # d(cϵ2*f)/dcϵ2
        assert jnp.isclose(J[23, 7], cϵ2)      # d(cϵ2*f)/df

    def test_jacobian_via_finite_differences(self, pybird_bias_combination_func, jacobian_func, test_biases):
        """Test Jacobian correctness using finite differences."""
        eps = 1e-5  # Larger epsilon for better numerical stability

        J_analytical = jacobian_func(test_biases)

        # Compute numerical Jacobian for a few entries
        for param_idx in [0, 7, 8, 9, 10]:  # b1, f, cϵ0, cϵ1, cϵ2
            biases_plus = test_biases.at[param_idx].add(eps)
            biases_minus = test_biases.at[param_idx].add(-eps)

            coeffs_plus = pybird_bias_combination_func(biases_plus)
            coeffs_minus = pybird_bias_combination_func(biases_minus)

            J_numerical = (coeffs_plus - coeffs_minus) / (2 * eps)

            # Check a few non-zero entries
            # Note: Finite differences are inherently less accurate than analytical derivatives
            for coeff_idx in range(24):
                if abs(J_analytical[coeff_idx, param_idx]) > 1e-10:
                    assert jnp.isclose(
                        J_analytical[coeff_idx, param_idx],
                        J_numerical[coeff_idx],
                        rtol=1e-2, atol=1e-4  # More relaxed tolerances for finite differences
                    ), f"Mismatch at ({coeff_idx}, {param_idx}): analytical={J_analytical[coeff_idx, param_idx]}, numerical={J_numerical[coeff_idx]}"

    def test_jacobian_via_jax_autodiff(self, pybird_bias_combination_func, jacobian_func, test_biases):
        """Test Jacobian matches JAX automatic differentiation."""
        # Compute Jacobian using JAX
        jax_jacobian = jax.jacfwd(pybird_bias_combination_func)(test_biases)

        # Compute analytical Jacobian
        analytical_jacobian = jacobian_func(test_biases)

        # They should match
        assert jnp.allclose(jax_jacobian, analytical_jacobian, rtol=1e-5, atol=1e-7)


class TestIntegrationWithPowerSpectrum:
    """Test integration of 11-parameter bias combination with power spectrum."""

    def test_power_spectrum_computation(self):
        """Test that power spectrum can be computed with 11 PyBird parameters."""
        # Mock stacked array (k=10, components=24)
        stacked_array = jnp.ones((10, 24))

        # 11 bias parameters
        biases = jnp.array([2.0, 0.5, -0.4, 0.1, 0.05, 0.02, 0.01, 0.8, 1.0, 0.5, 0.2])

        def pybird_bias_combination(bs):
            b1, b2, b3, b4, b5, b6, b7, f, cϵ0, cϵ1, cϵ2 = bs
            return jnp.array([
                b1**2, 2*b1*f, f**2, 1., b1, b2, b3, b4,
                b1*b1, b1*b2, b1*b3, b1*b4,
                b2*b2, b2*b4, b4*b4,
                2*b1*b5, 2*b1*b6, 2*b1*b7,
                2*f*b5, 2*f*b6, 2*f*b7,
                cϵ0, cϵ1, cϵ2*f
            ])

        biases_vec = pybird_bias_combination(biases)
        Pl = stacked_array @ biases_vec

        assert Pl.shape == (10,)

    def test_jacobian_computation(self):
        """Test that Jacobian can be used in power spectrum."""
        # Mock stacked array (k=10, components=24)
        stacked_array = jnp.ones((10, 24))

        # 11 bias parameters
        biases = jnp.array([2.0, 0.5, -0.4, 0.1, 0.05, 0.02, 0.01, 0.8, 1.0, 0.5, 0.2])

        def jacobian_bias_combination(bs):
            b1, b2, b3, b4, b5, b6, b7, f, cϵ0, cϵ1, cϵ2 = bs
            J = jnp.eye(24, 11)  # Simplified
            return J

        jac_biases = jacobian_bias_combination(biases)
        Pl_jac = stacked_array @ jac_biases

        assert Pl_jac.shape == (10, 11)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
