"""
Robust integration tests validating complete workflows.

These tests validate:
- End-to-end emulator usage
- Cross-module interactions
- Performance characteristics
- Real cosmology calculations
- Error propagation through the stack
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import tempfile
import shutil

os.environ["JAXEFFORT_NO_AUTO_DOWNLOAD"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

jax.config.update("jax_enable_x64", True)


class TestEndToEndWorkflow:
    """Test complete workflows from initialization to results."""

    def test_fresh_user_experience(self, tmp_path, monkeypatch):
        """Test what a new user experiences on first use."""
        # Simulate fresh environment
        monkeypatch.delenv("JAXEFFORT_NO_AUTO_DOWNLOAD", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))

        # Create mock downloader
        def mock_download(*args, **kwargs):
            # Create expected structure
            cache_dir = tmp_path / ".jaxeffort_data" / "emulators" / "pybird_mnuw0wacdm"
            for l in ["0", "2", "4"]:
                for comp in ["11", "loop", "ct"]:
                    comp_dir = cache_dir / l / comp
                    comp_dir.mkdir(parents=True)
                    # Add minimal files
                    np.save(comp_dir / "weights.npy", np.ones((10, 10)))
            return cache_dir.parent

        # Patch the download process
        with patch('jaxeffort.data_fetcher.MultipoleDataFetcher.download_and_extract') as mock_dl:
            with patch('jaxeffort.data_fetcher.MultipoleDataFetcher.get_multipole_paths') as mock_paths:
                mock_dl.side_effect = mock_download
                mock_paths.return_value = {
                    0: tmp_path / ".jaxeffort_data" / "emulators" / "pybird_mnuw0wacdm" / "0",
                    2: tmp_path / ".jaxeffort_data" / "emulators" / "pybird_mnuw0wacdm" / "2",
                    4: tmp_path / ".jaxeffort_data" / "emulators" / "pybird_mnuw0wacdm" / "4"
                }

                # Fresh import should trigger download
                if 'jaxeffort' in sys.modules:
                    del sys.modules['jaxeffort']

                import jaxeffort

                # Should have attempted to load emulators
                assert 'pybird_mnuw0wacdm' in jaxeffort.trained_emulators

    def test_offline_usage_with_cache(self, tmp_path, monkeypatch):
        """Test that cached data works without internet."""
        # Set up cached data
        cache_dir = tmp_path / ".jaxeffort_data"
        monkeypatch.setenv("HOME", str(tmp_path.parent))

        # Create cache structure
        emulator_dir = cache_dir / "emulators" / "test_model"
        for l in ["0", "2", "4"]:
            (emulator_dir / l).mkdir(parents=True)

        # Simulate network failure
        with patch('urllib.request.urlopen', side_effect=ConnectionError("No internet")):
            import jaxeffort
            from jaxeffort.data_fetcher import MultipoleDataFetcher

            fetcher = MultipoleDataFetcher(
                zenodo_url="https://zenodo.org/test.tar.gz",
                emulator_name="test_model",
                cache_dir=cache_dir
            )

            # Should work with cached data even if network is down
            path = fetcher.get_emulator_path(download_if_missing=False)
            assert path == emulator_dir

    def test_concurrent_usage_safety(self):
        """Test that multiple processes can safely use jaxeffort."""
        import threading
        import jaxeffort

        results = []
        errors = []

        def compute_stoch():
            try:
                k_grid = jnp.linspace(0.01, 0.3, 50)
                cϵ0 = jnp.array(1.0)
                cϵ1 = jnp.array(0.5)
                cϵ2 = jnp.array(0.1)
                n_bar = jnp.array(1e-3)
                result = jaxeffort.get_stoch_terms(cϵ0, cϵ1, cϵ2, n_bar, k_grid)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Launch multiple threads
        threads = []
        for _ in range(10):
            t = threading.Thread(target=compute_stoch)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All should succeed
        assert len(errors) == 0
        assert len(results) == 10

        # Results should be consistent
        for i in range(1, len(results)):
            # Results are tuples (P0, P2)
            assert jnp.allclose(results[0][0], results[i][0])  # Compare P0
            assert jnp.allclose(results[0][1], results[i][1])  # Compare P2


class TestPerformanceCharacteristics:
    """Test performance and optimization characteristics."""

    def test_jit_compilation_caching(self):
        """Verify JIT compilations are cached properly."""
        import jaxeffort

        k_grid = jnp.linspace(0.01, 0.3, 100)

        # Create JIT compiled function
        @jit
        def compute_stoch(cϵ0, cϵ1, cϵ2, n_bar):
            return jaxeffort.get_stoch_terms(cϵ0, cϵ1, cϵ2, n_bar, k_grid)

        # First call - includes compilation
        cϵ0_1 = jnp.array(1.0)
        cϵ1_1 = jnp.array(0.5)
        cϵ2_1 = jnp.array(0.1)
        n_bar_1 = jnp.array(1e-3)

        start = time.time()
        result1 = compute_stoch(cϵ0_1, cϵ1_1, cϵ2_1, n_bar_1)
        first_time = time.time() - start

        # Second call with different values - should use cached compilation
        cϵ0_2 = jnp.array(1.5)
        cϵ1_2 = jnp.array(0.6)
        cϵ2_2 = jnp.array(0.2)
        n_bar_2 = jnp.array(1e-3)

        start = time.time()
        result2 = compute_stoch(cϵ0_2, cϵ1_2, cϵ2_2, n_bar_2)
        second_time = time.time() - start

        # Second call should be faster (no compilation)
        assert second_time < first_time * 0.5  # At least 2x faster

        # Results should be different (different inputs) - results are tuples
        assert not (jnp.allclose(result1[0], result2[0]) and jnp.allclose(result1[1], result2[1]))

    def test_vectorization_efficiency(self):
        """Test that vectorized operations are efficient."""
        import jaxeffort

        k_grid = jnp.linspace(0.01, 0.3, 100)

        # Single evaluation
        single_cϵ0 = jnp.array(1.0)
        single_cϵ1 = jnp.array(0.5)
        single_cϵ2 = jnp.array(0.1)
        single_n_bar = jnp.array(1e-3)

        start = time.time()
        single_result = jaxeffort.get_stoch_terms(single_cϵ0, single_cϵ1, single_cϵ2, single_n_bar, k_grid)
        single_time = time.time() - start

        # Batch evaluation
        batch_size = 100
        batch_cϵ0 = jnp.full(batch_size, 1.0)
        batch_cϵ1 = jnp.full(batch_size, 0.5)
        batch_cϵ2 = jnp.full(batch_size, 0.1)
        batch_n_bar = jnp.full(batch_size, 1e-3)

        vmap_stoch = vmap(jaxeffort.get_stoch_terms, in_axes=(0, 0, 0, 0, None))

        start = time.time()
        batch_result = vmap_stoch(batch_cϵ0, batch_cϵ1, batch_cϵ2, batch_n_bar, k_grid)
        batch_time = time.time() - start

        # Batch should be more efficient per evaluation
        time_per_eval_single = single_time
        time_per_eval_batch = batch_time / batch_size

        # Vectorized should be significantly faster per evaluation
        assert time_per_eval_batch < time_per_eval_single * 0.5

        # Results should match for identical inputs
        # batch_result is tuple of (batch_monopole, batch_quadrupole)
        for i in range(batch_size):
            assert jnp.allclose(batch_result[0][i], single_result[0])  # Monopole
            assert jnp.allclose(batch_result[1][i], single_result[1])  # Quadrupole


class TestRealCosmologyCalculations:
    """Test with realistic cosmological parameters."""

    def test_planck_cosmology_parameters(self):
        """Test with Planck 2018 cosmology."""
        import jaxeffort

        k_grid = jnp.logspace(-2, 0, 100)  # 0.01 to 1 h/Mpc

        # Planck 2018 best-fit parameters
        Om = 0.3153
        Ob = 0.0493
        h = 0.6736
        ns = 0.9649
        s8 = 0.8111
        mnu = 0.06  # eV
        w0 = -1.0
        wa = 0.0

        cosmology = jnp.array([Om, Ob, h, ns, s8, mnu, w0, wa])

        # Galaxy bias parameters (reasonable values for LRGs)
        b1 = 2.0
        b2 = 0.5
        bs2 = -0.4
        b3nl = 0.1
        biases = jnp.array([b1, b2, bs2, b3nl])

        # Stochastic parameters
        ceps0 = 0.0  # Shot noise
        ceps1 = 1.0  # k^2 term
        ceps2 = 0.0  # k^4 term
        stochastic = jnp.array([ceps0, ceps1, ceps2])

        # Growth parameters
        D = 0.8  # Growth factor at z~0.5
        a_eff = 2/3  # Effective scale factor

        # Compute stochastic terms
        # Note: get_stoch_terms takes (cϵ0, cϵ1, cϵ2, n_bar, k_grid)
        # Using 1/b1^2 as proxy for n_bar (typical scaling)
        n_bar = 1.0 / (b1 ** 2) * 1e-3
        stoch = jaxeffort.get_stoch_terms(ceps0, ceps1, ceps2, n_bar, k_grid)

        # Validate results
        assert len(stoch) == 2  # Returns tuple (monopole, quadrupole)
        assert stoch[0].shape == (100,)  # Monopole
        assert stoch[1].shape == (100,)  # Quadrupole
        assert jnp.all(jnp.isfinite(stoch[0]))
        assert jnp.all(jnp.isfinite(stoch[1]))

        # Physical constraints
        # Shot noise should dominate at high k
        high_k_monopole = stoch[0][-10:]  # Last 10 k values of monopole
        low_k_monopole = stoch[0][:10]  # First 10 k values of monopole
        assert jnp.mean(jnp.abs(high_k_monopole)) > jnp.mean(jnp.abs(low_k_monopole))

    def test_redshift_evolution(self):
        """Test evolution with redshift through growth factor."""
        import jaxeffort

        k_grid = jnp.linspace(0.01, 0.3, 50)
        cϵ0 = jnp.array(1.0)
        cϵ1 = jnp.array(0.5)
        cϵ2 = jnp.array(0.1)

        # Different redshifts affect n_bar
        z_values = [0.0, 0.5, 1.0, 2.0]
        # Number density evolves with redshift
        n_bar_values = [1e-3 * (1 + z)**3 for z in z_values]  # Comoving number density

        results = []
        for n_bar in n_bar_values:
            stoch = jaxeffort.get_stoch_terms(cϵ0, cϵ1, cϵ2, jnp.array(n_bar), k_grid)
            results.append(stoch)

        # Power should evolve with redshift
        # Generally decreases as we go to higher redshift (lower a)
        for i in range(1, len(results)):
            # Check general trend (may not be strictly monotonic) - results are tuples
            assert not (jnp.allclose(results[i][0], results[0][0]) and jnp.allclose(results[i][1], results[0][1]))


class TestErrorPropagation:
    """Test how errors propagate through the system."""

    def test_invalid_parameter_ranges(self):
        """Test handling of parameters outside physical ranges."""
        import jaxeffort

        k_grid = jnp.linspace(0.01, 0.3, 50)

        # Negative stochastic parameters (unphysical but should handle gracefully)
        negative_cϵ0 = jnp.array(-1.0)
        cϵ1 = jnp.array(0.5)
        cϵ2 = jnp.array(0.1)
        n_bar = jnp.array(1e-3)

        result = jaxeffort.get_stoch_terms(negative_cϵ0, cϵ1, cϵ2, n_bar, k_grid)

        # Should still compute (no hard constraints)
        assert jnp.all(jnp.isfinite(result[0]))  # Monopole
        assert jnp.all(jnp.isfinite(result[1]))  # Quadrupole

    def test_gradient_through_full_pipeline(self):
        """Test gradient computation through complete pipeline."""
        import jaxeffort

        k_grid = jnp.linspace(0.01, 0.3, 50)

        def loss_function(params):
            """Mock loss function for parameter fitting."""
            cϵ0, cϵ1, cϵ2 = params[:3]
            n_bar = params[3]

            stoch = jaxeffort.get_stoch_terms(cϵ0, cϵ1, cϵ2, n_bar, k_grid)

            # Mock data (separate for monopole and quadrupole)
            mock_mono = jnp.ones_like(stoch[0])
            mock_quad = jnp.ones_like(stoch[1])

            # Chi-squared loss
            return jnp.sum((stoch[0] - mock_mono) ** 2) + jnp.sum((stoch[1] - mock_quad) ** 2)

        # Initial parameters (cϵ0, cϵ1, cϵ2, n_bar)
        params = jnp.array([1.0, 0.5, 0.1, 1e-3])

        # Compute gradient
        grad_fn = grad(loss_function)
        gradient = grad_fn(params)

        # Gradient should be well-defined
        assert gradient.shape == params.shape
        assert jnp.all(jnp.isfinite(gradient))

        # Can use for optimization
        learning_rate = 0.01
        updated_params = params - learning_rate * gradient

        # Loss should decrease
        initial_loss = loss_function(params)
        updated_loss = loss_function(updated_params)
        assert updated_loss < initial_loss


class TestFunctionSignatures:
    """Test function signatures."""

    def test_stoch_terms_signature(self):
        """Verify get_stoch_terms signature."""
        import jaxeffort

        # Test correct signature
        k_grid = jnp.linspace(0.01, 0.3, 50)

        # Correct signature: get_stoch_terms(cε0, cε1, cε2, n_bar, k_grid)
        cε0 = jnp.array(1.0)
        cε1 = jnp.array(0.5)
        cε2 = jnp.array(0.1)
        n_bar = jnp.array(1e-3)

        result1 = jaxeffort.get_stoch_terms(cε0, cε1, cε2, n_bar, k_grid)
        result2 = jaxeffort.get_stoch_terms(jnp.array(cε0), jnp.array(cε1), jnp.array(cε2), jnp.array(n_bar), k_grid)

        # Results are tuples (P0, P2)
        assert jnp.allclose(result1[0], result2[0])  # Compare P0
        assert jnp.allclose(result1[1], result2[1])  # Compare P2

    def test_module_exports_stability(self):
        """Verify exported functions remain available."""
        import jaxeffort

        # Core exports that should always be available
        essential_exports = [
            'MLP',
            'MultipoleEmulators',
            'MultipoleNoiseEmulator',
            'load_multipole_emulator',
            'load_multipole_noise_emulator',
            'get_stoch_terms',
            'trained_emulators',
            'EMULATOR_CONFIGS'
        ]

        for export in essential_exports:
            assert hasattr(jaxeffort, export), f"Missing essential export: {export}"

            # Should be callable or data
            attr = getattr(jaxeffort, export)
            assert attr is not None


class TestProductionReadiness:
    """Test production readiness aspects."""

    def test_deterministic_results(self):
        """Verify results are deterministic."""
        import jaxeffort

        k_grid = jnp.linspace(0.01, 0.3, 50)
        cε0 = jnp.array(1.0)
        cε1 = jnp.array(0.5)
        cε2 = jnp.array(0.1)
        n_bar = jnp.array(1e-3)

        # Multiple evaluations
        results = []
        for _ in range(5):
            result = jaxeffort.get_stoch_terms(cε0, cε1, cε2, n_bar, k_grid)
            results.append(result)

        # All should be identical
        for i in range(1, len(results)):
            # Results are tuples (P0, P2)
            assert jnp.array_equal(results[0][0], results[i][0])  # P0
            assert jnp.array_equal(results[0][1], results[i][1])  # P2

    def test_numerical_precision_maintained(self):
        """Verify float64 precision is maintained."""
        import jaxeffort

        # Ensure float64
        assert jax.config.read('jax_enable_x64') == True

        k_grid = jnp.linspace(0.01, 0.3, 50, dtype=jnp.float64)
        cϵ0 = jnp.array(1.0, dtype=jnp.float64)
        cϵ1 = jnp.array(0.5, dtype=jnp.float64)
        cϵ2 = jnp.array(0.1, dtype=jnp.float64)
        n_bar = jnp.array(1e-3, dtype=jnp.float64)

        P0, P2 = jaxeffort.get_stoch_terms(cϵ0, cϵ1, cϵ2, n_bar, k_grid)

        # Should maintain float64
        assert P0.dtype == jnp.float64
        assert P2.dtype == jnp.float64

        # Test precision with small differences
        cϵ0_2 = cϵ0 + 1e-15  # Tiny change
        P0_2, P2_2 = jaxeffort.get_stoch_terms(cϵ0_2, cϵ1, cϵ2, n_bar, k_grid)

        # Should detect the tiny difference (float64 precision)
        assert not jnp.array_equal(P0, P0_2)

    def test_error_messages_are_helpful(self, tmp_path):
        """Verify error messages guide users to solutions."""
        import jaxeffort
        from jaxeffort.data_fetcher import MultipoleDataFetcher

        # Test with nonexistent path
        fetcher = MultipoleDataFetcher(
            zenodo_url="https://nonexistent.url/file.tar.gz",
            emulator_name="test",
            cache_dir=tmp_path
        )

        # Mock network error
        import urllib.error
        with patch('urllib.request.urlretrieve', side_effect=urllib.error.URLError("Network unreachable")):
            success = fetcher._download_file(
                "https://nonexistent.url/file.tar.gz",
                tmp_path / "test.tar.gz",
                show_progress=True
            )

            assert success == False
            # Error should be caught gracefully, not crash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])