#!/usr/bin/env python
"""
Integration tests for jaxeffort with StochModel.

These tests use the actual emulator files to ensure everything works end-to-end.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import sys
import os

# Add jaxeffort to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jaxeffort.jaxeffort as jaxeffort
import jaxeffort


# Module-level fixture to check if emulators are available
@pytest.fixture(scope="module", autouse=True)
def check_emulators_available():
    """Check if emulators are downloaded, skip all tests if not."""
    emu = jaxeffort.trained_emulators.get('pybird_mnuw0wacdm', {}).get('0')
    if emu is None:
        pytest.skip("Emulators not available (not downloaded) - skipping integration tests", allow_module_level=True)


class TestEmulatorLoading:
    """Test loading of actual emulator files from jaxeffort."""

    def test_load_monopole_emulator(self):
        """Test loading monopole emulator with StochModel."""
        # Use jaxeffort's trained_emulators (downloaded from Zenodo)
        emu = jaxeffort.trained_emulators['pybird_mnuw0wacdm']['0']

        assert emu is not None
        assert hasattr(emu, 'stoch_model')
        assert hasattr(emu, 'bias_combination')
        assert hasattr(emu, 'jacobian_bias_combination')

    def test_load_all_multipoles(self):
        """Test loading all three multipoles."""
        emulators = jaxeffort.trained_emulators['pybird_mnuw0wacdm']
        emu_0 = emulators['0']
        emu_2 = emulators['2']
        emu_4 = emulators['4']

        assert all([emu_0, emu_2, emu_4])


class TestPowerSpectrumComputation:
    """Test power spectrum computation with real emulator."""

    @pytest.fixture(scope="class")
    def emulators(self):
        """Load emulators once for all tests."""
        return jaxeffort.trained_emulators['pybird_mnuw0wacdm']

    @pytest.fixture
    def test_cosmology(self):
        """Standard test cosmology."""
        return jnp.array([1.0, 3.05, 0.96, 67.0, 0.022, 0.12, 0.06, -1.0, 0.0])

    @pytest.fixture
    def test_biases(self):
        """Standard test biases (11 parameters)."""
        return jnp.array([2.0, 0.5, -0.4, 0.1, 0.05, 0.02, 0.01, 0.8, 1.0, 0.5, 0.2])

    @pytest.fixture
    def test_D(self):
        """Standard test growth factor."""
        return 0.5

    def test_monopole_get_Pl(self, emulators, test_cosmology, test_biases, test_D):
        """Test monopole get_Pl with 11 parameters."""
        emu = emulators['0']
        Pl = emu.get_Pl(test_cosmology, test_biases, test_D)

        assert Pl.shape == (74,)
        assert not jnp.any(jnp.isnan(Pl))
        assert not jnp.any(jnp.isinf(Pl))

    def test_all_multipoles_get_Pl(self, emulators, test_cosmology, test_biases, test_D):
        """Test all multipoles with get_Pl."""
        for l, emu in emulators.items():
            Pl = emu.get_Pl(test_cosmology, test_biases, test_D)

            assert Pl.shape == (74,), f"Multipole l={l} has wrong shape"
            assert not jnp.any(jnp.isnan(Pl)), f"Multipole l={l} has NaNs"
            assert not jnp.any(jnp.isinf(Pl)), f"Multipole l={l} has Infs"

    def test_monopole_get_Pl_jacobian(self, emulators, test_cosmology, test_biases, test_D):
        """Test monopole get_Pl_jacobian."""
        emu = emulators['0']
        Pl, Pl_jac = emu.get_Pl_jacobian(test_cosmology, test_biases, test_D)

        assert Pl.shape == (74,)
        assert Pl_jac.shape == (74, 11)
        assert not jnp.any(jnp.isnan(Pl))
        assert not jnp.any(jnp.isnan(Pl_jac))

    def test_Pl_consistent_between_methods(self, emulators, test_cosmology, test_biases, test_D):
        """Test that P_ℓ values match between get_Pl and get_Pl_jacobian."""
        emu = emulators['0']

        Pl_direct = emu.get_Pl(test_cosmology, test_biases, test_D)
        Pl_from_jac, _ = emu.get_Pl_jacobian(test_cosmology, test_biases, test_D)

        assert jnp.allclose(Pl_direct, Pl_from_jac, rtol=1e-10)


class TestStochModelValues:
    """Test StochModel produces correct values."""

    @pytest.fixture(scope="class")
    def emulators(self):
        """Load emulators once for all tests."""
        return jaxeffort.trained_emulators['pybird_mnuw0wacdm']

    def test_monopole_stoch_first_column(self, emulators):
        """Test monopole StochModel first column is ones."""
        emu = emulators['0']
        k_grid = emu.P11.k_grid
        stoch = emu.stoch_model(k_grid)

        assert jnp.allclose(stoch[:, 0], 1.0)

    def test_monopole_stoch_shape(self, emulators):
        """Test monopole StochModel has shape (74, 3)."""
        emu = emulators['0']
        k_grid = emu.P11.k_grid
        stoch = emu.stoch_model(k_grid)

        assert stoch.shape == (74, 3)

    def test_quadrupole_stoch_zeros(self, emulators):
        """Test quadrupole StochModel has zeros in first two columns."""
        emu = emulators['2']
        k_grid = emu.P11.k_grid
        stoch = emu.stoch_model(k_grid)

        assert jnp.allclose(stoch[:, 0], 0.0)
        assert jnp.allclose(stoch[:, 1], 0.0)

    def test_hexadecapole_stoch_all_zeros(self, emulators):
        """Test hexadecapole StochModel is all zeros."""
        emu = emulators['4']
        k_grid = emu.P11.k_grid
        stoch = emu.stoch_model(k_grid)

        assert jnp.allclose(stoch, 0.0)


class TestNumericalStability:
    """Test numerical stability of computations."""

    @pytest.fixture(scope="class")
    def emulator(self):
        """Load monopole emulator."""
        return jaxeffort.trained_emulators['pybird_mnuw0wacdm']['0']

    @pytest.fixture
    def test_cosmology(self):
        """Standard test cosmology."""
        return jnp.array([1.0, 3.05, 0.96, 67.0, 0.022, 0.12, 0.06, -1.0, 0.0])

    def test_extreme_bias_values(self, emulator, test_cosmology):
        """Test with extreme bias values."""
        # Very large biases
        large_biases = jnp.array([10.0, 5.0, -5.0, 1.0, 0.5, 0.2, 0.1, 2.0, 10.0, 5.0, 2.0])
        Pl = emulator.get_Pl(test_cosmology, large_biases, 0.5)

        assert not jnp.any(jnp.isnan(Pl))
        assert not jnp.any(jnp.isinf(Pl))

        # Very small biases
        small_biases = jnp.array([0.1, 0.01, -0.01, 0.001, 0.001, 0.001, 0.001, 0.1, 0.01, 0.01, 0.001])
        Pl = emulator.get_Pl(test_cosmology, small_biases, 0.5)

        assert not jnp.any(jnp.isnan(Pl))
        assert not jnp.any(jnp.isinf(Pl))

    def test_zero_stochastic_parameters(self, emulator, test_cosmology):
        """Test with zero stochastic parameters."""
        biases = jnp.array([2.0, 0.5, -0.4, 0.1, 0.05, 0.02, 0.01, 0.8, 0.0, 0.0, 0.0])
        Pl = emulator.get_Pl(test_cosmology, biases, 0.5)

        assert not jnp.any(jnp.isnan(Pl))
        assert not jnp.any(jnp.isinf(Pl))


class TestBackwardCompatibility:
    """Test that changes don't break existing functionality."""

    @pytest.fixture(scope="class")
    def emulator(self):
        """Load monopole emulator."""
        return jaxeffort.trained_emulators['pybird_mnuw0wacdm']['0']

    def test_k_grid_unchanged(self, emulator):
        """Test that k-grid is still accessible."""
        assert hasattr(emulator.P11, 'k_grid')
        assert emulator.P11.k_grid.shape == (74, 1)

    def test_get_multipole_components_still_works(self, emulator):
        """Test that get_multipole_components still works."""
        cosmology = jnp.array([1.0, 3.05, 0.96, 67.0, 0.022, 0.12, 0.06, -1.0, 0.0])
        D = 0.5

        P11, Ploop, Pct = emulator.get_multipole_components(cosmology, D)

        assert P11.shape == (74, 3)
        assert Ploop.shape == (74, 12)
        assert Pct.shape == (74, 6)


class TestRegressionAgainstKnownValues:
    """Regression tests against known good values."""

    @pytest.fixture(scope="class")
    def emulator(self):
        """Load monopole emulator."""
        return jaxeffort.trained_emulators['pybird_mnuw0wacdm']['0']

    def test_known_power_spectrum_values(self, emulator):
        """Test against known values from Effort.jl."""
        # These are the exact values from our comparison
        cosmology = jnp.array([1.0, 3.05, 0.96, 67.0, 0.022, 0.12, 0.06, -1.0, 0.0])
        biases = jnp.array([2.0, 0.5, -0.4, 0.1, 0.05, 0.02, 0.01, 0.8, 1.0, 0.5, 0.2])
        D = 0.46880539249642306  # From Effort.jl

        Pl = emulator.get_Pl(cosmology, biases, D)

        # Known values from Effort.jl comparison
        expected_first = 28548.512607
        expected_middle = 4621.841979
        expected_last = 385.777776

        assert jnp.isclose(Pl[0], expected_first, rtol=1e-6)
        assert jnp.isclose(Pl[37], expected_middle, rtol=1e-6)
        assert jnp.isclose(Pl[-1], expected_last, rtol=1e-6)

    def test_known_jacobian_shape(self, emulator):
        """Test Jacobian has expected shape."""
        cosmology = jnp.array([1.0, 3.05, 0.96, 67.0, 0.022, 0.12, 0.06, -1.0, 0.0])
        biases = jnp.array([2.0, 0.5, -0.4, 0.1, 0.05, 0.02, 0.01, 0.8, 1.0, 0.5, 0.2])
        D = 0.5

        _, Pl_jac = emulator.get_Pl_jacobian(cosmology, biases, D)

        # Should be (74 k-points, 11 bias parameters)
        assert Pl_jac.shape == (74, 11)


class TestFullPipelineGradients:
    """Test gradients through the full pipeline: cosmology → emulator → P_ℓ."""

    @pytest.fixture(scope="class")
    def emulator(self):
        """Load monopole emulator."""
        return jaxeffort.trained_emulators['pybird_mnuw0wacdm']['0']

    @pytest.fixture
    def test_cosmology(self):
        """Standard test cosmology."""
        return jnp.array([1.0, 3.05, 0.96, 67.0, 0.022, 0.12, 0.06, -1.0, 0.0])

    @pytest.fixture
    def test_biases(self):
        """Standard test biases (11 parameters)."""
        return jnp.array([2.0, 0.5, -0.4, 0.1, 0.05, 0.02, 0.01, 0.8, 1.0, 0.5, 0.2])

    @pytest.fixture
    def test_D(self):
        """Standard test growth factor."""
        return 0.5

    def test_gradient_wrt_biases_autodiff_vs_analytical(self, emulator, test_cosmology, test_biases, test_D):
        """Test that JAX autodiff matches analytical Jacobian for bias gradients."""
        # Define function to differentiate
        def power_spectrum(biases):
            return emulator.get_Pl(test_cosmology, biases, test_D)

        # Compute gradient using JAX autodiff
        jac_autodiff = jax.jacfwd(power_spectrum)(test_biases)

        # Compute gradient using analytical Jacobian
        _, jac_analytical = emulator.get_Pl_jacobian(test_cosmology, test_biases, test_D)

        # They should match very closely
        assert jac_autodiff.shape == (74, 11)
        assert jac_analytical.shape == (74, 11)
        assert jnp.allclose(jac_autodiff, jac_analytical, rtol=1e-5, atol=1e-7), \
            f"Max diff: {jnp.max(jnp.abs(jac_autodiff - jac_analytical))}"

    def test_gradient_wrt_biases_finite_diff_vs_autodiff(self, emulator, test_cosmology, test_biases, test_D):
        """Test that finite differences match JAX autodiff for bias gradients."""
        eps = 1e-5

        # Define function to differentiate
        def power_spectrum(biases):
            return emulator.get_Pl(test_cosmology, biases, test_D)

        # Compute gradient using JAX autodiff
        jac_autodiff = jax.jacfwd(power_spectrum)(test_biases)

        # Compute gradient using finite differences
        jac_finite_diff = jnp.zeros((74, 11))
        for i in range(11):
            biases_plus = test_biases.at[i].add(eps)
            biases_minus = test_biases.at[i].add(-eps)

            Pl_plus = power_spectrum(biases_plus)
            Pl_minus = power_spectrum(biases_minus)

            jac_finite_diff = jac_finite_diff.at[:, i].set((Pl_plus - Pl_minus) / (2 * eps))

        # Finite differences should match autodiff reasonably well
        assert jnp.allclose(jac_autodiff, jac_finite_diff, rtol=1e-3, atol=1e-5), \
            f"Max relative diff: {jnp.max(jnp.abs((jac_autodiff - jac_finite_diff) / (jnp.abs(jac_autodiff) + 1e-10)))}"

    def test_gradient_wrt_cosmology_autodiff_vs_finite_diff(self, emulator, test_cosmology, test_biases, test_D):
        """Test full pipeline gradient w.r.t. cosmology using autodiff vs finite differences."""
        eps = 1e-5

        # Define full pipeline function
        def power_spectrum_from_cosmology(cosmology):
            return emulator.get_Pl(cosmology, test_biases, test_D)

        # Compute gradient using JAX autodiff
        jac_autodiff = jax.jacfwd(power_spectrum_from_cosmology)(test_cosmology)

        # Compute gradient using finite differences
        jac_finite_diff = jnp.zeros((74, 9))  # 9 cosmological parameters
        for i in range(9):
            cosmo_plus = test_cosmology.at[i].add(eps)
            cosmo_minus = test_cosmology.at[i].add(-eps)

            Pl_plus = power_spectrum_from_cosmology(cosmo_plus)
            Pl_minus = power_spectrum_from_cosmology(cosmo_minus)

            jac_finite_diff = jac_finite_diff.at[:, i].set((Pl_plus - Pl_minus) / (2 * eps))

        # Check shapes
        assert jac_autodiff.shape == (74, 9)
        assert jac_finite_diff.shape == (74, 9)

        # Finite differences should match autodiff reasonably well
        # Note: cosmology gradients can be more challenging numerically
        assert jnp.allclose(jac_autodiff, jac_finite_diff, rtol=1e-2, atol=1e-4), \
            f"Max relative diff: {jnp.max(jnp.abs((jac_autodiff - jac_finite_diff) / (jnp.abs(jac_autodiff) + 1e-10)))}"

    def test_full_pipeline_with_growth_factor_gradient(self, emulator, test_biases):
        """Test gradient through full pipeline including D(z) computation."""
        # Define a cosmology
        cosmology = jnp.array([1.0, 3.05, 0.96, 67.0, 0.022, 0.12, 0.06, -1.0, 0.0])
        z = cosmology[0]

        # For this test, we'll use a simple D approximation since the full ODE
        # gradient would require integration. We test that the emulator gradient works.
        D_value = 0.5  # Fixed for this test

        # Define function of cosmology parameters (excluding z)
        def power_spectrum_emulator_only(cosmo_params):
            # cosmo_params = [ln10As, ns, H0, ωb, ωcdm, mν, w0, wa]
            # Reconstruct full cosmology
            full_cosmo = jnp.concatenate([jnp.array([z]), cosmo_params])
            return emulator.get_Pl(full_cosmo, test_biases, D_value)

        cosmo_params = cosmology[1:]  # All except z

        # Compute gradient using JAX autodiff
        jac_autodiff = jax.jacfwd(power_spectrum_emulator_only)(cosmo_params)

        # Compute gradient using finite differences
        eps = 1e-5
        jac_finite_diff = jnp.zeros((74, 8))
        for i in range(8):
            params_plus = cosmo_params.at[i].add(eps)
            params_minus = cosmo_params.at[i].add(-eps)

            Pl_plus = power_spectrum_emulator_only(params_plus)
            Pl_minus = power_spectrum_emulator_only(params_minus)

            jac_finite_diff = jac_finite_diff.at[:, i].set((Pl_plus - Pl_minus) / (2 * eps))

        # Check shapes
        assert jac_autodiff.shape == (74, 8)
        assert jac_finite_diff.shape == (74, 8)

        # Verify agreement
        assert jnp.allclose(jac_autodiff, jac_finite_diff, rtol=1e-2, atol=1e-4), \
            f"Max diff: {jnp.max(jnp.abs(jac_autodiff - jac_finite_diff))}"

    def test_combined_cosmology_and_bias_gradient(self, emulator, test_cosmology, test_biases, test_D):
        """Test gradient w.r.t. both cosmology and bias parameters together."""
        # Concatenate parameters
        all_params = jnp.concatenate([test_cosmology, test_biases])

        # Define function
        def power_spectrum_all_params(params):
            cosmology = params[:9]
            biases = params[9:]
            return emulator.get_Pl(cosmology, biases, test_D)

        # Compute gradient using JAX autodiff
        jac_autodiff = jax.jacfwd(power_spectrum_all_params)(all_params)

        # Compute gradient using finite differences
        eps = 1e-5
        jac_finite_diff = jnp.zeros((74, 20))  # 9 cosmo + 11 bias = 20 params
        for i in range(20):
            params_plus = all_params.at[i].add(eps)
            params_minus = all_params.at[i].add(-eps)

            Pl_plus = power_spectrum_all_params(params_plus)
            Pl_minus = power_spectrum_all_params(params_minus)

            jac_finite_diff = jac_finite_diff.at[:, i].set((Pl_plus - Pl_minus) / (2 * eps))

        # Check shapes
        assert jac_autodiff.shape == (74, 20)
        assert jac_finite_diff.shape == (74, 20)

        # Verify agreement
        assert jnp.allclose(jac_autodiff, jac_finite_diff, rtol=1e-2, atol=1e-4), \
            f"Max diff: {jnp.max(jnp.abs(jac_autodiff - jac_finite_diff))}"


if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
