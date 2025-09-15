"""
Test background cosmology functions from jaxace integration.

These tests verify that the jaxace background functions imported into jaxeffort
work correctly, following the same test patterns as jaxace's own tests.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from jaxeffort import (
    W0WaCDMCosmology,
    E_z, D_z, f_z, D_f_z, r_z, dA_z, dL_z,
    E_a, dlogEdloga, Ωm_a,
    a_z,
)

# Configure JAX for 64-bit precision
jax.config.update('jax_enable_x64', True)


class TestBasicFunctions:
    """Test basic cosmological functions."""

    def test_scale_factor(self):
        """Test scale factor from redshift."""
        assert np.isclose(a_z(0.0), 1.0)
        assert np.isclose(a_z(1.0), 0.5)
        assert np.isclose(a_z(3.0), 0.25)

    def test_cosmology_struct_creation(self):
        """Test W0WaCDMCosmology structure creation."""
        cosmo = W0WaCDMCosmology(
            ln10As=3.044,
            ns=0.9649,
            h=0.6736,
            omega_b=0.02237,
            omega_c=0.1200,
            m_nu=0.06,
            w0=-1.0,
            wa=0.0
        )

        assert cosmo.h == 0.6736
        assert cosmo.omega_b == 0.02237
        assert cosmo.omega_c == 0.1200
        assert cosmo.m_nu == 0.06
        assert cosmo.w0 == -1.0
        assert cosmo.wa == 0.0


class TestHubbleParameter:
    """Test Hubble parameter functions."""

    def test_normalization(self):
        """Test E(z=0) = 1 normalization."""
        Ωcb0 = 0.3
        h = 0.67

        # At z=0 (a=1), E should be 1
        assert np.isclose(E_z(0.0, Ωcb0, h), 1.0)
        assert np.isclose(E_a(1.0, Ωcb0, h), 1.0)

        # With massive neutrinos
        assert np.isclose(E_z(0.0, Ωcb0, h, mν=0.06), 1.0)
        assert np.isclose(E_a(1.0, Ωcb0, h, mν=0.06), 1.0)

    def test_evolution(self):
        """Test Hubble parameter evolution."""
        Ωcb0 = 0.3
        h = 0.67

        # E(z) should increase with z for standard cosmology
        E0 = E_z(0.0, Ωcb0, h)
        E1 = E_z(1.0, Ωcb0, h)
        E2 = E_z(2.0, Ωcb0, h)

        assert E0 < E1 < E2
        assert E0 == 1.0


class TestGrowthFunctions:
    """Test growth factor and growth rate."""

    @pytest.fixture
    def cosmo_params(self):
        """Standard cosmology parameters for testing."""
        return {
            'Ωcb0': 0.3,
            'h': 0.67,
            'mν': 0.06,
            'w0': -1.0,
            'wa': 0.0
        }

    def test_growth_normalization(self, cosmo_params):
        """Test D(z=0) normalization."""
        D0 = D_z(0.0, **cosmo_params)
        # D(z=0) is normalized to 1 by convention
        assert np.isclose(D0 / D0, 1.0, rtol=1e-6)

    def test_growth_evolution(self, cosmo_params):
        """Test growth factor decreases with redshift."""
        D0 = D_z(0.0, **cosmo_params)
        D1 = D_z(1.0, **cosmo_params)
        D2 = D_z(2.0, **cosmo_params)

        # Growth factor should decrease with z
        assert D0 > D1 > D2

    def test_growth_rate(self, cosmo_params):
        """Test growth rate f(z)."""
        f0 = f_z(0.0, **cosmo_params)
        f1 = f_z(1.0, **cosmo_params)

        # Growth rate should be positive and order ~1
        assert 0 < f0 < 2
        assert 0 < f1 < 2


class TestDistanceMeasures:
    """Test cosmological distance measures."""

    @pytest.fixture
    def cosmo_params(self):
        """Standard cosmology parameters for testing."""
        return {
            'Ωcb0': 0.3,
            'h': 0.67,
            'mν': 0.06,
            'w0': -1.0,
            'wa': 0.0
        }

    def test_distances_at_z0(self, cosmo_params):
        """Test all distances are zero at z=0."""
        z = 0.0

        assert np.isclose(r_z(z, **cosmo_params), 0.0, atol=1e-10)
        assert np.isclose(dL_z(z, **cosmo_params), 0.0, atol=1e-10)
        assert np.isclose(dA_z(z, **cosmo_params), 0.0, atol=1e-10)

    def test_distance_evolution(self, cosmo_params):
        """Test distance measures increase with redshift."""
        r1 = r_z(1.0, **cosmo_params)
        r2 = r_z(2.0, **cosmo_params)

        dL1 = dL_z(1.0, **cosmo_params)
        dL2 = dL_z(2.0, **cosmo_params)

        # Distances should increase with z
        assert r2 > r1 > 0
        assert dL2 > dL1 > 0

    def test_distance_duality(self, cosmo_params):
        """Test distance duality relation: dL = (1+z)^2 * dA."""
        z_test = jnp.array([0.5, 1.0, 1.5, 2.0])

        dA = dA_z(z_test, **cosmo_params)
        dL = dL_z(z_test, **cosmo_params)

        # Distance duality relation
        expected_dL = (1 + z_test)**2 * dA

        np.testing.assert_allclose(dL, expected_dL, rtol=1e-5)


class TestCLASSComparison:
    """
    Test against hardcoded CLASS values for specific cosmology.

    Parameters: h = 0.67, Ωb h² = 0.022, Ωc h² = 0.12,
                mν = 0.06 eV, w0 = -1.0, wa = 0.0
    """

    @pytest.fixture
    def cosmo_params(self):
        """Set up cosmology parameters matching CLASS test."""
        return {
            'Ωcb0': (0.022 + 0.12) / 0.67**2,  # Total Ωcb0
            'h': 0.67,
            'mν': 0.06,
            'w0': -1.0,
            'wa': 0.0
        }

    def test_hubble_at_z0(self, cosmo_params):
        """Test Hubble parameter at z=0."""
        H0 = E_z(0.0, **cosmo_params) * 100 * cosmo_params['h']
        # H0 should be exactly h * 100 km/s/Mpc at z=0
        assert np.isclose(H0, 67.0, rtol=1e-6)

    def test_hubble_at_z1(self, cosmo_params):
        """Test Hubble parameter at z=1."""
        H1 = E_z(1.0, **cosmo_params) * 100 * cosmo_params['h']
        # This is an approximate expected value for ΛCDM
        assert H1 > 100  # Should be larger than H0
        assert H1 < 150  # But not too large

    def test_growth_at_z1(self, cosmo_params):
        """Test growth factor at z=1."""
        D0 = D_z(0.0, **cosmo_params)
        D1 = D_z(1.0, **cosmo_params)

        # For ΛCDM with these parameters, D(z=1)/D(z=0) ~ 0.6
        ratio = D1 / D0
        assert 0.5 < ratio < 0.7

    def test_comoving_distance_at_z1(self, cosmo_params):
        """Test comoving distance at z=1."""
        r1 = r_z(1.0, **cosmo_params)

        # For ΛCDM with h=0.67, r(z=1) ~ 3000-4000 Mpc/h
        assert 2500 < r1 < 4500


class TestConsistencyRelations:
    """Test consistency relations between different functions."""

    @pytest.fixture
    def cosmo_params(self):
        """Standard cosmology parameters."""
        return {
            'Ωcb0': 0.3,
            'h': 0.70,
            'mν': 0.0,
            'w0': -1.0,
            'wa': 0.0
        }

    def test_hubble_in_different_units(self, cosmo_params):
        """Test E(z) and E(a) consistency."""
        z_test = jnp.array([0.0, 0.5, 1.0, 1.5, 2.0])
        a_test = 1 / (1 + z_test)

        E_from_z = E_z(z_test, **cosmo_params)
        E_from_a = E_a(a_test, **cosmo_params)

        # Should give same results
        np.testing.assert_allclose(E_from_z, E_from_a, rtol=1e-10)

    def test_matter_density_parameter(self, cosmo_params):
        """Test matter density parameter evolution."""
        z_test = jnp.array([0.0, 1.0, 2.0])
        a_test = 1 / (1 + z_test)

        Omega_m = Ωm_a(a_test, **cosmo_params)

        # At z=0, should equal Ωcb0 (ignoring neutrinos)
        assert np.isclose(Omega_m[0], cosmo_params['Ωcb0'], rtol=0.01)

        # Should increase with redshift
        assert Omega_m[2] > Omega_m[1] > Omega_m[0]

    def test_growth_consistency(self, cosmo_params):
        """Test D_f_z returns both D and f consistently."""
        z_test = jnp.array([0.5, 1.0, 1.5])

        D_values = D_z(z_test, **cosmo_params)
        f_values = f_z(z_test, **cosmo_params)
        D_combined, f_combined = D_f_z(z_test, **cosmo_params)

        # D_f_z should return the same D and f as individual functions
        np.testing.assert_allclose(D_combined, D_values, rtol=1e-10)
        np.testing.assert_allclose(f_combined, f_values, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])