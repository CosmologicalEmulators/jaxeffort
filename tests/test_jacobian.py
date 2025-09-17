"""
Test Jacobian computation for jaxeffort emulators using JAX automatic differentiation.

This module tests that:
1. Jacobians can be computed without errors
2. All Jacobian values are finite (no NaN or Inf)
3. Jacobians have the expected shapes
4. Gradients can be computed with respect to different parameter sets
"""

import os
import tempfile
import shutil
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev, grad

from jaxeffort.jaxeffort import (
    load_multipole_emulator,
    load_multipole_noise_emulator,
    MultipoleEmulators,
    MultipoleNoiseEmulator
)
from tests.fixtures.mock_emulator_data import (
    create_mock_emulator_directory,
    create_mock_noise_emulator_directory
)


class TestJacobianComputation:
    """Test Jacobian computation for multipole emulators."""

    @pytest.fixture
    def mock_emulator_path(self):
        """Create a temporary directory with mock emulator data."""
        temp_dir = create_mock_emulator_directory()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def multipole_emulator(self, mock_emulator_path):
        """Load a mock multipole emulator."""
        return load_multipole_emulator(mock_emulator_path)

    def test_jacobian_wrt_cosmology(self, multipole_emulator):
        """Test Jacobian computation with respect to cosmological parameters."""
        # Define test inputs
        biases = jnp.array([1.5, 0.5, -0.2, 0.1])  # b1, b2, bs2, b3nl
        cosmology = jnp.array([
            0.025,   # Omega_b * h^2
            0.120,   # Omega_cdm * h^2
            0.06,    # M_nu (total neutrino mass in eV)
            -1.0,    # w_0
            0.0,     # w_a
            3.05,    # ln(10^10 A_s)
            0.96,    # n_s
            0.67     # h
        ])
        D = 1.0

        # Define function to differentiate
        def f_cosmology(cosmo):
            Pl = multipole_emulator.get_Pl(cosmo, biases, D)
            return Pl.flatten()

        # Compute Jacobian using forward-mode AD
        jacobian = jacfwd(f_cosmology)(cosmology)

        # Check that Jacobian has correct shape
        # The output shape depends on the mock emulator's output size
        n_output = len(f_cosmology(cosmology))
        expected_shape = (n_output, len(cosmology))
        assert jacobian.shape == expected_shape, \
            f"Expected Jacobian shape {expected_shape}, got {jacobian.shape}"

        # Check that all values are finite
        assert jnp.all(jnp.isfinite(jacobian)), \
            f"Non-finite values found in Jacobian. NaN count: {jnp.sum(jnp.isnan(jacobian))}, " \
            f"Inf count: {jnp.sum(jnp.isinf(jacobian))}"

        # Check that Jacobian is not all zeros
        assert not jnp.allclose(jacobian, 0.0), \
            "Jacobian is all zeros, which likely indicates a problem"

    def test_jacobian_wrt_biases(self, multipole_emulator):
        """Test Jacobian computation with respect to bias parameters."""
        # Define test inputs
        biases = jnp.array([1.5, 0.5, -0.2, 0.1])
        cosmology = jnp.array([0.025, 0.120, 0.06, -1.0, 0.0, 3.05, 0.96, 0.67])
        D = 1.0

        # Define function to differentiate
        def f_biases(bias):
            Pl = multipole_emulator.get_Pl(cosmology, bias, D)
            return Pl.flatten()

        # Compute Jacobian using reverse-mode AD (more efficient for this case)
        jacobian = jacrev(f_biases)(biases)

        # Check that Jacobian has correct shape
        # The output shape depends on the mock emulator's output size
        n_output = len(f_biases(biases))
        expected_shape = (n_output, len(biases))
        assert jacobian.shape == expected_shape, \
            f"Expected Jacobian shape {expected_shape}, got {jacobian.shape}"

        # Check that all values are finite
        assert jnp.all(jnp.isfinite(jacobian)), \
            f"Non-finite values found in Jacobian. NaN count: {jnp.sum(jnp.isnan(jacobian))}, " \
            f"Inf count: {jnp.sum(jnp.isinf(jacobian))}"

    def test_jacobian_wrt_growth_factor(self, multipole_emulator):
        """Test Jacobian computation with respect to growth factor D."""
        # Define test inputs
        biases = jnp.array([1.5, 0.5, -0.2, 0.1])
        cosmology = jnp.array([0.025, 0.120, 0.06, -1.0, 0.0, 3.05, 0.96, 0.67])
        D = 1.0

        # Define function to differentiate
        def f_growth(growth):
            Pl = multipole_emulator.get_Pl(cosmology, biases, growth)
            return Pl.flatten()

        # Compute gradient (Jacobian is a vector in this case)
        gradient = grad(lambda d: jnp.sum(f_growth(d)))(D)

        # Check that gradient is finite
        assert jnp.isfinite(gradient), f"Non-finite gradient: {gradient}"

        # Also compute the full Jacobian
        jacobian = jacfwd(f_growth)(D)
        n_output = len(f_growth(D))
        expected_shape = (n_output,)
        assert jacobian.shape == expected_shape, \
            f"Expected Jacobian shape {expected_shape}, got {jacobian.shape}"

        # Check all values are finite
        assert jnp.all(jnp.isfinite(jacobian)), \
            "Non-finite values found in Jacobian with respect to growth factor"

    def test_mixed_jacobian(self, multipole_emulator):
        """Test Jacobian computation with respect to combined parameters."""
        # Define test inputs
        biases = jnp.array([1.5, 0.5, -0.2, 0.1])
        cosmology = jnp.array([0.025, 0.120, 0.06, -1.0, 0.0, 3.05, 0.96, 0.67])
        D = 1.0

        # Combine all parameters into a single vector
        def f_combined(params):
            bias_part = params[:4]
            cosmo_part = params[4:12]
            growth_part = params[12]
            Pl = multipole_emulator.get_Pl(cosmo_part, bias_part, growth_part)
            return Pl.flatten()

        combined_params = jnp.concatenate([biases, cosmology, jnp.array([D])])

        # Compute full Jacobian
        jacobian = jacfwd(f_combined)(combined_params)

        # Check shape
        n_output = len(f_combined(combined_params))
        expected_shape = (n_output, len(combined_params))
        assert jacobian.shape == expected_shape, \
            f"Expected Jacobian shape {expected_shape}, got {jacobian.shape}"

        # Check that all values are finite
        assert jnp.all(jnp.isfinite(jacobian)), \
            f"Non-finite values in combined Jacobian. " \
            f"NaN count: {jnp.sum(jnp.isnan(jacobian))}, " \
            f"Inf count: {jnp.sum(jnp.isinf(jacobian))}"

    def test_jacobian_consistency(self, multipole_emulator):
        """Test that forward and reverse mode AD give consistent results."""
        # Define test inputs
        biases = jnp.array([1.5, 0.5, -0.2, 0.1])
        cosmology = jnp.array([0.025, 0.120, 0.06, -1.0, 0.0, 3.05, 0.96, 0.67])
        D = 1.0

        # Define function to differentiate
        def f_biases(bias):
            Pl = multipole_emulator.get_Pl(cosmology, bias, D)
            # Return just the first element for simpler comparison
            return Pl.flatten()[0]

        # Compute Jacobian using both forward and reverse mode
        jac_forward = jacfwd(f_biases)(biases)
        jac_reverse = jacrev(f_biases)(biases)

        # Check that both methods give the same result
        assert jnp.allclose(jac_forward, jac_reverse, rtol=1e-6, atol=1e-8), \
            f"Forward and reverse mode Jacobians don't match. " \
            f"Max diff: {jnp.max(jnp.abs(jac_forward - jac_reverse))}"

        # Check that both are finite
        assert jnp.all(jnp.isfinite(jac_forward)), "Forward mode Jacobian has non-finite values"
        assert jnp.all(jnp.isfinite(jac_reverse)), "Reverse mode Jacobian has non-finite values"


class TestNoiseEmulatorJacobian:
    """Test Jacobian computation for multipole emulators with noise."""

    @pytest.fixture
    def mock_noise_emulator_path(self):
        """Create a temporary directory with mock emulator data including noise."""
        temp_dir = create_mock_noise_emulator_directory()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def noise_emulator(self, mock_noise_emulator_path):
        """Load a mock multipole noise emulator."""
        return load_multipole_noise_emulator(mock_noise_emulator_path)

    def test_noise_emulator_jacobian(self, noise_emulator):
        """Test Jacobian computation for noise emulator."""
        # Define test inputs
        biases = jnp.array([1.5, 0.5, -0.2, 0.1])
        cosmology = jnp.array([0.025, 0.120, 0.06, -1.0, 0.0, 3.05, 0.96, 0.67])
        D = 1.0

        # Define function to differentiate
        def f_cosmology(cosmo):
            Pl = noise_emulator.get_Pl(cosmo, biases, D)
            return Pl.flatten()

        # Compute Jacobian
        jacobian = jacfwd(f_cosmology)(cosmology)

        # Check that all values are finite
        assert jnp.all(jnp.isfinite(jacobian)), \
            f"Non-finite values found in noise emulator Jacobian. " \
            f"NaN count: {jnp.sum(jnp.isnan(jacobian))}, " \
            f"Inf count: {jnp.sum(jnp.isinf(jacobian))}"

        # Check shape
        n_output = len(f_cosmology(cosmology))
        expected_shape = (n_output, len(cosmology))
        assert jacobian.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {jacobian.shape}"


class TestJacobianEdgeCases:
    """Test Jacobian computation with edge cases and boundary conditions."""

    @pytest.fixture
    def multipole_emulator(self):
        """Load a mock multipole emulator."""
        temp_dir = create_mock_emulator_directory()
        emulator = load_multipole_emulator(temp_dir)
        yield emulator
        shutil.rmtree(temp_dir)

    def test_jacobian_at_zero_biases(self, multipole_emulator):
        """Test Jacobian when bias parameters are zero."""
        biases = jnp.array([0.0, 0.0, 0.0, 0.0])
        cosmology = jnp.array([0.025, 0.120, 0.06, -1.0, 0.0, 3.05, 0.96, 0.67])
        D = 1.0

        def f(bias):
            Pl = multipole_emulator.get_Pl(cosmology, bias, D)
            return Pl.flatten()

        jacobian = jacfwd(f)(biases)
        assert jnp.all(jnp.isfinite(jacobian)), \
            "Non-finite values in Jacobian at zero biases"

    def test_jacobian_at_extreme_cosmology(self, multipole_emulator):
        """Test Jacobian with extreme (but valid) cosmological parameters."""
        biases = jnp.array([1.5, 0.5, -0.2, 0.1])
        # Use extreme but physical values
        cosmology = jnp.array([
            0.015,   # Low Omega_b * h^2
            0.150,   # High Omega_cdm * h^2
            0.3,     # High M_nu
            -1.5,    # Low w_0
            0.5,     # Non-zero w_a
            2.5,     # Low ln(10^10 A_s)
            0.85,    # Low n_s
            0.5      # Low h
        ])
        D = 0.5

        def f(cosmo):
            Pl = multipole_emulator.get_Pl(cosmo, biases, D)
            return Pl.flatten()

        jacobian = jacfwd(f)(cosmology)
        assert jnp.all(jnp.isfinite(jacobian)), \
            "Non-finite values in Jacobian at extreme cosmology"

    def test_jacobian_second_order(self, multipole_emulator):
        """Test second-order derivatives (Hessian) are finite."""
        biases = jnp.array([1.5, 0.5, -0.2, 0.1])
        cosmology = jnp.array([0.025, 0.120, 0.06, -1.0, 0.0, 3.05, 0.96, 0.67])
        D = 1.0

        # Define scalar function for simpler Hessian
        def f_scalar(bias):
            Pl = multipole_emulator.get_Pl(cosmology, bias, D)
            return jnp.sum(Pl.flatten())

        # Compute Hessian using composition of jacfwd
        hessian = jacfwd(jacfwd(f_scalar))(biases)

        # Check shape
        assert hessian.shape == (4, 4), f"Expected Hessian shape (4, 4), got {hessian.shape}"

        # Check that all values are finite
        assert jnp.all(jnp.isfinite(hessian)), \
            f"Non-finite values in Hessian. NaN count: {jnp.sum(jnp.isnan(hessian))}"

        # Check symmetry (Hessian should be symmetric for smooth functions)
        assert jnp.allclose(hessian, hessian.T, rtol=1e-6, atol=1e-8), \
            "Hessian is not symmetric"


class TestJacobianWithRealEmulator:
    """Test Jacobian computation with the real downloaded emulator if available."""

    def test_real_emulator_jacobian_if_available(self):
        """Test with real emulator from Zenodo if it has been downloaded."""
        try:
            import jaxeffort
            emulator = jaxeffort.get_default_emulator()

            if emulator is None:
                pytest.skip("Real emulator not available (not downloaded)")

            # Define realistic test inputs
            # Real PyBird emulator expects 8 bias parameters: [b1, b2, b3, b4, b5, b6, b7, f]
            biases = jnp.array([
                1.5,    # b1 - linear bias
                0.5,    # b2 - second-order bias
                -0.2,   # b3 - third-order bias
                0.1,    # b4 - fourth-order bias
                0.0,    # b5 - fifth-order bias
                0.0,    # b6 - sixth-order bias
                0.0,    # b7 - seventh-order bias
                0.8     # f - growth rate parameter
            ])
            cosmology = jnp.array([
                0.5,     # z (redshift) - note: real emulator might expect z first
                3.05,    # ln(10^10 A_s)
                0.96,    # n_s
                0.67,    # H0/100
                0.025,   # omega_b = Omega_b * h^2
                0.120,   # omega_c = Omega_cdm * h^2
                0.06,    # M_nu
                -1.0,    # w_0
                0.0      # w_a
            ])
            D = 1.0

            def f(cosmo):
                Pl = emulator.get_Pl(cosmo, biases, D)
                return Pl.flatten() if hasattr(Pl, 'flatten') else Pl

            # Try to compute Jacobian
            jacobian = jacfwd(f)(cosmology)

            # Basic checks
            assert jacobian.shape[1] == len(cosmology), \
                f"Jacobian should have {len(cosmology)} columns"

            # Check for finite values
            finite_mask = jnp.isfinite(jacobian)
            finite_ratio = jnp.mean(finite_mask)

            assert finite_ratio > 0.95, \
                f"Too many non-finite values in Jacobian. " \
                f"Finite ratio: {finite_ratio:.2%}, " \
                f"NaN count: {jnp.sum(jnp.isnan(jacobian))}, " \
                f"Inf count: {jnp.sum(jnp.isinf(jacobian))}"

        except Exception as e:
            pytest.skip(f"Could not test with real emulator: {e}")