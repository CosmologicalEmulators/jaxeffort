"""
Test Velocileptors emulators (LPT and REPT).

This module tests that:
1. Velocileptors emulators can be loaded and called
2. All outputs are finite (no NaN or Inf)
3. Output shapes are correct
4. Jacobians can be computed using JAX automatic differentiation
5. Gradients with respect to cosmology and bias parameters work correctly
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev

from tests.fixtures.sample_cosmologies import (
    get_test_cosmology_array,
    get_test_biases_velocileptors,
)


class TestVelocileptorsLPTEmulator:
    """Test Velocileptors LPT emulator."""

    @pytest.fixture
    def lpt_emulator(self):
        """Load Velocileptors LPT emulator."""
        import jaxeffort
        emulator = jaxeffort.trained_emulators.get('velocileptors_lpt_mnuw0wacdm', {})
        if not emulator or emulator.get('0') is None:
            pytest.skip("Velocileptors LPT emulator not available (not downloaded)")
        return emulator

    def test_lpt_emulator_loads(self, lpt_emulator):
        """Test that LPT emulator loads successfully."""
        # Check all multipoles are loaded
        assert '0' in lpt_emulator
        assert '2' in lpt_emulator
        assert '4' in lpt_emulator

        # Check monopole has all components
        monopole = lpt_emulator['0']
        assert monopole is not None
        assert monopole.P11 is not None
        assert monopole.Ploop is not None
        assert monopole.Pct is not None
        assert monopole.bias_combination is not None
        assert monopole.stoch_model is not None

    def test_lpt_forward_pass_monopole(self, lpt_emulator):
        """Test forward pass through LPT monopole emulator."""
        monopole = lpt_emulator['0']

        # Create test inputs
        cosmology = get_test_cosmology_array()
        biases = get_test_biases_velocileptors()
        D = 0.8

        # Compute power spectrum
        Pl = monopole.get_Pl(cosmology, biases, D)

        # Check output is valid
        assert Pl is not None
        assert Pl.ndim == 1
        assert len(Pl) > 0

        # Check all values are finite (no NaN or Inf)
        assert jnp.all(jnp.isfinite(Pl)), \
            f"Non-finite values in LPT monopole output. NaN: {jnp.sum(jnp.isnan(Pl))}, Inf: {jnp.sum(jnp.isinf(Pl))}"

    def test_lpt_forward_pass_all_multipoles(self, lpt_emulator):
        """Test forward pass through all LPT multipoles."""
        cosmology = get_test_cosmology_array()
        biases = get_test_biases_velocileptors()
        D = 0.8

        for l in ['0', '2', '4']:
            multipole = lpt_emulator[l]
            Pl = multipole.get_Pl(cosmology, biases, D)

            # Check output
            assert Pl is not None, f"LPT multipole l={l} returned None"
            assert jnp.all(jnp.isfinite(Pl)), \
                f"Non-finite values in LPT l={l}. NaN: {jnp.sum(jnp.isnan(Pl))}, Inf: {jnp.sum(jnp.isinf(Pl))}"

    def test_lpt_components_output(self, lpt_emulator):
        """Test that LPT components have correct structure."""
        monopole = lpt_emulator['0']
        cosmology = get_test_cosmology_array()
        D = 0.8

        # Get components
        P11, Ploop, Pct = monopole.get_multipole_components(cosmology, D)

        # Check all components are returned
        assert P11 is not None
        assert Ploop is not None
        assert Pct is not None

        # Check shapes are consistent
        assert P11.shape[0] == Ploop.shape[0]
        assert Ploop.shape[0] == Pct.shape[0]

        # Check all values are finite
        assert jnp.all(jnp.isfinite(P11)), "Non-finite values in P11"
        assert jnp.all(jnp.isfinite(Ploop)), "Non-finite values in Ploop"
        assert jnp.all(jnp.isfinite(Pct)), "Non-finite values in Pct"

    def test_lpt_jacobian_analytical(self, lpt_emulator):
        """Test analytical Jacobian computation for LPT."""
        monopole = lpt_emulator['0']

        # Check if analytical Jacobian is available
        if monopole.jacobian_bias_combination is None:
            pytest.skip("Analytical Jacobian not available for LPT")

        cosmology = get_test_cosmology_array()
        biases = get_test_biases_velocileptors()
        D = 0.8

        # Compute Jacobian
        Pl, Pl_jac = monopole.get_Pl_jacobian(cosmology, biases, D)

        # Check outputs
        assert Pl is not None
        assert Pl_jac is not None

        # Check shapes
        n_k = len(Pl)
        n_bias = len(biases)
        assert Pl_jac.shape == (n_k, n_bias), \
            f"Expected Jacobian shape ({n_k}, {n_bias}), got {Pl_jac.shape}"

        # Check all values are finite
        assert jnp.all(jnp.isfinite(Pl)), "Non-finite values in Pl"
        assert jnp.all(jnp.isfinite(Pl_jac)), \
            f"Non-finite values in Jacobian. NaN: {jnp.sum(jnp.isnan(Pl_jac))}, Inf: {jnp.sum(jnp.isinf(Pl_jac))}"

    def test_lpt_jacfwd_wrt_cosmology(self, lpt_emulator):
        """Test JAX jacfwd with respect to cosmological parameters for LPT."""
        monopole = lpt_emulator['0']
        cosmology = get_test_cosmology_array()
        biases = get_test_biases_velocileptors()
        D = 0.8

        # Define function to differentiate
        def f_cosmology(cosmo):
            return monopole.get_Pl(cosmo, biases, D)

        # Compute Jacobian using forward-mode AD
        jacobian = jacfwd(f_cosmology)(cosmology)

        # Check shape
        n_output = len(f_cosmology(cosmology))
        n_cosmo = len(cosmology)
        expected_shape = (n_output, n_cosmo)
        assert jacobian.shape == expected_shape, \
            f"Expected Jacobian shape {expected_shape}, got {jacobian.shape}"

        # Check all values are finite
        assert jnp.all(jnp.isfinite(jacobian)), \
            f"Non-finite values in cosmology Jacobian. NaN: {jnp.sum(jnp.isnan(jacobian))}, Inf: {jnp.sum(jnp.isinf(jacobian))}"

        # Check Jacobian is not all zeros
        assert not jnp.allclose(jacobian, 0.0, atol=1e-10), \
            "Jacobian is all zeros, which likely indicates a problem"

    def test_lpt_jacfwd_wrt_biases(self, lpt_emulator):
        """Test JAX jacfwd with respect to bias parameters for LPT."""
        monopole = lpt_emulator['0']
        cosmology = get_test_cosmology_array()
        biases = get_test_biases_velocileptors()
        D = 0.8

        # Define function to differentiate
        def f_biases(b):
            return monopole.get_Pl(cosmology, b, D)

        # Compute Jacobian using forward-mode AD
        jacobian = jacfwd(f_biases)(biases)

        # Check shape
        n_output = len(f_biases(biases))
        n_bias = len(biases)
        expected_shape = (n_output, n_bias)
        assert jacobian.shape == expected_shape, \
            f"Expected Jacobian shape {expected_shape}, got {jacobian.shape}"

        # Check all values are finite
        assert jnp.all(jnp.isfinite(jacobian)), \
            f"Non-finite values in bias Jacobian. NaN: {jnp.sum(jnp.isnan(jacobian))}, Inf: {jnp.sum(jnp.isinf(jacobian))}"

        # Check Jacobian is not all zeros
        assert not jnp.allclose(jacobian, 0.0, atol=1e-10), \
            "Jacobian is all zeros, which likely indicates a problem"

    def test_lpt_jacrev_wrt_cosmology(self, lpt_emulator):
        """Test JAX jacrev with respect to cosmological parameters for LPT."""
        monopole = lpt_emulator['0']
        cosmology = get_test_cosmology_array()
        biases = get_test_biases_velocileptors()
        D = 0.8

        # Define function to differentiate
        def f_cosmology(cosmo):
            return monopole.get_Pl(cosmo, biases, D)

        # Compute Jacobian using reverse-mode AD
        jacobian = jacrev(f_cosmology)(cosmology)

        # Check shape
        n_output = len(f_cosmology(cosmology))
        n_cosmo = len(cosmology)
        expected_shape = (n_output, n_cosmo)
        assert jacobian.shape == expected_shape, \
            f"Expected Jacobian shape {expected_shape}, got {jacobian.shape}"

        # Check all values are finite
        assert jnp.all(jnp.isfinite(jacobian)), \
            f"Non-finite values in cosmology Jacobian (jacrev). NaN: {jnp.sum(jnp.isnan(jacobian))}, Inf: {jnp.sum(jnp.isinf(jacobian))}"


class TestVelocileptorsREPTEmulator:
    """Test Velocileptors REPT emulator."""

    @pytest.fixture
    def rept_emulator(self):
        """Load Velocileptors REPT emulator."""
        import jaxeffort
        emulator = jaxeffort.trained_emulators.get('velocileptors_rept_mnuw0wacdm', {})
        if not emulator or emulator.get('0') is None:
            pytest.skip("Velocileptors REPT emulator not available (not downloaded)")
        return emulator

    def test_rept_emulator_loads(self, rept_emulator):
        """Test that REPT emulator loads successfully."""
        # Check all multipoles are loaded
        assert '0' in rept_emulator
        assert '2' in rept_emulator
        assert '4' in rept_emulator

        # Check monopole has all components
        monopole = rept_emulator['0']
        assert monopole is not None
        assert monopole.P11 is not None
        assert monopole.Ploop is not None
        assert monopole.Pct is not None
        assert monopole.bias_combination is not None
        assert monopole.stoch_model is not None

    def test_rept_forward_pass_monopole(self, rept_emulator):
        """Test forward pass through REPT monopole emulator."""
        monopole = rept_emulator['0']

        # Create test inputs
        cosmology = get_test_cosmology_array()
        biases = get_test_biases_velocileptors()
        D = 0.8

        # Compute power spectrum
        Pl = monopole.get_Pl(cosmology, biases, D)

        # Check output is valid
        assert Pl is not None
        assert Pl.ndim == 1
        assert len(Pl) > 0

        # Check all values are finite (no NaN or Inf)
        assert jnp.all(jnp.isfinite(Pl)), \
            f"Non-finite values in REPT monopole output. NaN: {jnp.sum(jnp.isnan(Pl))}, Inf: {jnp.sum(jnp.isinf(Pl))}"

    def test_rept_forward_pass_all_multipoles(self, rept_emulator):
        """Test forward pass through all REPT multipoles."""
        cosmology = get_test_cosmology_array()
        biases = get_test_biases_velocileptors()
        D = 0.8

        for l in ['0', '2', '4']:
            multipole = rept_emulator[l]
            Pl = multipole.get_Pl(cosmology, biases, D)

            # Check output
            assert Pl is not None, f"REPT multipole l={l} returned None"
            assert jnp.all(jnp.isfinite(Pl)), \
                f"Non-finite values in REPT l={l}. NaN: {jnp.sum(jnp.isnan(Pl))}, Inf: {jnp.sum(jnp.isinf(Pl))}"

    def test_rept_components_output(self, rept_emulator):
        """Test that REPT components have correct structure."""
        monopole = rept_emulator['0']
        cosmology = get_test_cosmology_array()
        D = 0.8

        # Get components
        P11, Ploop, Pct = monopole.get_multipole_components(cosmology, D)

        # Check all components are returned
        assert P11 is not None
        assert Ploop is not None
        assert Pct is not None

        # Check shapes are consistent
        assert P11.shape[0] == Ploop.shape[0]
        assert Ploop.shape[0] == Pct.shape[0]

        # Check all values are finite
        assert jnp.all(jnp.isfinite(P11)), "Non-finite values in P11"
        assert jnp.all(jnp.isfinite(Ploop)), "Non-finite values in Ploop"
        assert jnp.all(jnp.isfinite(Pct)), "Non-finite values in Pct"

    def test_rept_jacobian_analytical(self, rept_emulator):
        """Test analytical Jacobian computation for REPT."""
        monopole = rept_emulator['0']

        # Check if analytical Jacobian is available
        if monopole.jacobian_bias_combination is None:
            pytest.skip("Analytical Jacobian not available for REPT")

        cosmology = get_test_cosmology_array()
        biases = get_test_biases_velocileptors()
        D = 0.8

        # Compute Jacobian
        Pl, Pl_jac = monopole.get_Pl_jacobian(cosmology, biases, D)

        # Check outputs
        assert Pl is not None
        assert Pl_jac is not None

        # Check shapes
        n_k = len(Pl)
        n_bias = len(biases)
        assert Pl_jac.shape == (n_k, n_bias), \
            f"Expected Jacobian shape ({n_k}, {n_bias}), got {Pl_jac.shape}"

        # Check all values are finite
        assert jnp.all(jnp.isfinite(Pl)), "Non-finite values in Pl"
        assert jnp.all(jnp.isfinite(Pl_jac)), \
            f"Non-finite values in Jacobian. NaN: {jnp.sum(jnp.isnan(Pl_jac))}, Inf: {jnp.sum(jnp.isinf(Pl_jac))}"

    def test_rept_jacfwd_wrt_cosmology(self, rept_emulator):
        """Test JAX jacfwd with respect to cosmological parameters for REPT."""
        monopole = rept_emulator['0']
        cosmology = get_test_cosmology_array()
        biases = get_test_biases_velocileptors()
        D = 0.8

        # Define function to differentiate
        def f_cosmology(cosmo):
            return monopole.get_Pl(cosmo, biases, D)

        # Compute Jacobian using forward-mode AD
        jacobian = jacfwd(f_cosmology)(cosmology)

        # Check shape
        n_output = len(f_cosmology(cosmology))
        n_cosmo = len(cosmology)
        expected_shape = (n_output, n_cosmo)
        assert jacobian.shape == expected_shape, \
            f"Expected Jacobian shape {expected_shape}, got {jacobian.shape}"

        # Check all values are finite
        assert jnp.all(jnp.isfinite(jacobian)), \
            f"Non-finite values in cosmology Jacobian. NaN: {jnp.sum(jnp.isnan(jacobian))}, Inf: {jnp.sum(jnp.isinf(jacobian))}"

        # Check Jacobian is not all zeros
        assert not jnp.allclose(jacobian, 0.0, atol=1e-10), \
            "Jacobian is all zeros, which likely indicates a problem"

    def test_rept_jacfwd_wrt_biases(self, rept_emulator):
        """Test JAX jacfwd with respect to bias parameters for REPT."""
        monopole = rept_emulator['0']
        cosmology = get_test_cosmology_array()
        biases = get_test_biases_velocileptors()
        D = 0.8

        # Define function to differentiate
        def f_biases(b):
            return monopole.get_Pl(cosmology, b, D)

        # Compute Jacobian using forward-mode AD
        jacobian = jacfwd(f_biases)(biases)

        # Check shape
        n_output = len(f_biases(biases))
        n_bias = len(biases)
        expected_shape = (n_output, n_bias)
        assert jacobian.shape == expected_shape, \
            f"Expected Jacobian shape {expected_shape}, got {jacobian.shape}"

        # Check all values are finite
        assert jnp.all(jnp.isfinite(jacobian)), \
            f"Non-finite values in bias Jacobian. NaN: {jnp.sum(jnp.isnan(jacobian))}, Inf: {jnp.sum(jnp.isinf(jacobian))}"

        # Check Jacobian is not all zeros
        assert not jnp.allclose(jacobian, 0.0, atol=1e-10), \
            "Jacobian is all zeros, which likely indicates a problem"

    def test_rept_jacrev_wrt_cosmology(self, rept_emulator):
        """Test JAX jacrev with respect to cosmological parameters for REPT."""
        monopole = rept_emulator['0']
        cosmology = get_test_cosmology_array()
        biases = get_test_biases_velocileptors()
        D = 0.8

        # Define function to differentiate
        def f_cosmology(cosmo):
            return monopole.get_Pl(cosmo, biases, D)

        # Compute Jacobian using reverse-mode AD
        jacobian = jacrev(f_cosmology)(cosmology)

        # Check shape
        n_output = len(f_cosmology(cosmology))
        n_cosmo = len(cosmology)
        expected_shape = (n_output, n_cosmo)
        assert jacobian.shape == expected_shape, \
            f"Expected Jacobian shape {expected_shape}, got {jacobian.shape}"

        # Check all values are finite
        assert jnp.all(jnp.isfinite(jacobian)), \
            f"Non-finite values in cosmology Jacobian (jacrev). NaN: {jnp.sum(jnp.isnan(jacobian))}, Inf: {jnp.sum(jnp.isinf(jacobian))}"


class TestVelocileptorsComparison:
    """Compare LPT and REPT emulators."""

    @pytest.fixture
    def both_emulators(self):
        """Load both LPT and REPT emulators."""
        import jaxeffort
        lpt = jaxeffort.trained_emulators.get('velocileptors_lpt_mnuw0wacdm', {})
        rept = jaxeffort.trained_emulators.get('velocileptors_rept_mnuw0wacdm', {})

        if not lpt or lpt.get('0') is None or not rept or rept.get('0') is None:
            pytest.skip("Both Velocileptors emulators not available")

        return {'lpt': lpt, 'rept': rept}

    def test_different_outputs(self, both_emulators):
        """Test that LPT and REPT produce different outputs."""
        lpt = both_emulators['lpt']['0']
        rept = both_emulators['rept']['0']

        cosmology = get_test_cosmology_array()
        biases = get_test_biases_velocileptors()
        D = 0.8

        # Get outputs
        Pl_lpt = lpt.get_Pl(cosmology, biases, D)
        Pl_rept = rept.get_Pl(cosmology, biases, D)

        # Check that outputs are different (they use different theory)
        # Allow them to have different shapes (different k-grids)
        if Pl_lpt.shape == Pl_rept.shape:
            assert not jnp.allclose(Pl_lpt, Pl_rept, rtol=0.01), \
                "LPT and REPT outputs are too similar - they should differ"

    def test_both_have_same_k_grid_size(self, both_emulators):
        """Test that both emulators have consistent k-grids within themselves."""
        for name, emulator in both_emulators.items():
            monopole = emulator['0']
            k_grid = monopole.P11.k_grid

            # Just check it exists and has reasonable size
            assert len(k_grid) > 0, f"{name} has empty k-grid"
            assert len(k_grid) < 1000, f"{name} has suspiciously large k-grid"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
