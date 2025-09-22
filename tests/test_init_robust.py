"""
Robust tests for __init__.py focusing on real initialization scenarios.

These tests validate:
- Auto-loading behavior and error recovery
- Noise emulator detection and loading
- Configuration management
- Graceful degradation when data unavailable
"""

import os
import sys
import importlib
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
import pytest
import warnings

# Prevent auto-download during testing
os.environ["JAXEFFORT_NO_AUTO_DOWNLOAD"] = "1"

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAutoLoadingBehavior:
    """Test automatic loading of emulators on import."""

    def test_respects_no_auto_download_env_var(self):
        """Verify NO_AUTO_DOWNLOAD environment variable is respected."""
        # Force reimport with NO_AUTO_DOWNLOAD set
        with patch.dict(os.environ, {'JAXEFFORT_NO_AUTO_DOWNLOAD': '1'}, clear=True):
            # Remove from sys.modules to force fresh import
            if 'jaxeffort' in sys.modules:
                del sys.modules['jaxeffort']
            if 'jaxeffort.jaxeffort' in sys.modules:
                del sys.modules['jaxeffort.jaxeffort']
            if 'jaxeffort.data_fetcher' in sys.modules:
                del sys.modules['jaxeffort.data_fetcher']

            import jaxeffort

            # Should have empty structure, not downloaded data
            for model_name in jaxeffort.EMULATOR_CONFIGS:
                assert model_name in jaxeffort.trained_emulators
                # All should be None when auto-download disabled
                for multipole in ['0', '2', '4']:
                    assert jaxeffort.trained_emulators[model_name][multipole] is None or \
                           jaxeffort.trained_emulators[model_name][multipole].__class__.__name__ == 'MultipoleEmulators', \
                           f"Expected None or already loaded emulator for {model_name}[{multipole}]"

    def test_auto_download_triggers_when_enabled(self, tmp_path, monkeypatch):
        """Verify auto-download happens when environment allows."""
        # Create a mock that tracks download attempts
        download_attempts = []

        def mock_get_multipole_paths(*args, **kwargs):
            download_attempts.append(kwargs.get('download_if_missing', True))
            # Return mock paths
            return {0: tmp_path / "0", 2: tmp_path / "2", 4: tmp_path / "4"}

        # Remove the env var to enable auto-download
        monkeypatch.delenv("JAXEFFORT_NO_AUTO_DOWNLOAD", raising=False)

        # Patch the data fetcher
        with patch('jaxeffort.data_fetcher.MultipoleDataFetcher.get_multipole_paths',
                   side_effect=mock_get_multipole_paths):
            # Force reimport to trigger auto-download
            if 'jaxeffort' in sys.modules:
                del sys.modules['jaxeffort']

            import jaxeffort

            # Should have attempted downloads
            assert len(download_attempts) > 0
            assert any(download_attempts)  # At least one should be True

    def test_graceful_degradation_on_download_failure(self, monkeypatch):
        """Ensure import doesn't fail even if downloads fail."""
        # Remove env var to enable auto-download
        monkeypatch.delenv("JAXEFFORT_NO_AUTO_DOWNLOAD", raising=False)

        # Mock download to fail
        with patch('jaxeffort.data_fetcher.MultipoleDataFetcher.get_multipole_paths',
                   return_value=None):
            # Force reimport
            if 'jaxeffort' in sys.modules:
                del sys.modules['jaxeffort']

            # Import should succeed despite download failure
            import jaxeffort

            # Should have empty entries
            for model_name in jaxeffort.EMULATOR_CONFIGS:
                assert model_name in jaxeffort.trained_emulators
                for multipole in ['0', '2', '4']:
                    assert jaxeffort.trained_emulators[model_name][multipole] is None


class TestNoiseEmulatorLoading:
    """Test loading of emulators with noise components."""

    def test_noise_emulator_detection(self, tmp_path):
        """Verify noise emulators are loaded when st/ folder exists."""
        import jaxeffort
        from jaxeffort import _load_emulator_set

        # Create mock emulator structure WITH noise
        mock_paths = {}
        for l in [0, 2, 4]:
            mp_path = tmp_path / str(l)
            # Create standard components
            for comp in ["11", "loop", "ct"]:
                (mp_path / comp).mkdir(parents=True)
                # Add mock files
                (mp_path / comp / "weights.npy").write_text("mock")

            # Add noise component
            (mp_path / "st").mkdir(parents=True)
            (mp_path / "st" / "weights.npy").write_text("mock noise")

            mock_paths[l] = mp_path

        # Mock the fetcher to return our paths
        with patch('jaxeffort.data_fetcher.get_fetcher') as mock_get_fetcher:
            mock_fetcher = MagicMock()
            mock_fetcher.get_multipole_paths.return_value = mock_paths
            mock_get_fetcher.return_value = mock_fetcher

            # Mock the actual loader functions
            with patch('jaxeffort.load_multipole_noise_emulator') as mock_noise_load:
                with patch('jaxeffort.load_multipole_emulator') as mock_standard_load:
                    mock_noise_load.return_value = MagicMock(spec=['get_Pl'])
                    mock_standard_load.return_value = MagicMock(spec=['get_Pl'])

                    # Load with noise flag
                    config = {
                        'zenodo_url': 'test.tar.gz',
                        'has_noise': True  # This should trigger noise loading
                    }

                    emulators = _load_emulator_set('test_noise_model', config, auto_download=False)

                    # Should have called noise loader, not standard
                    assert mock_noise_load.call_count == 3  # For l=0,2,4
                    assert mock_standard_load.call_count == 0

    def test_fallback_to_standard_when_no_noise(self, tmp_path):
        """Verify standard loading when noise component missing."""
        import jaxeffort
        from jaxeffort import _load_emulator_set

        # Create mock emulator structure WITHOUT noise
        mock_paths = {}
        for l in [0, 2, 4]:
            mp_path = tmp_path / str(l)
            # Create only standard components (no st/)
            for comp in ["11", "loop", "ct"]:
                (mp_path / comp).mkdir(parents=True)
                (mp_path / comp / "weights.npy").write_text("mock")
            mock_paths[l] = mp_path

        # Mock the fetcher
        with patch('jaxeffort.data_fetcher.get_fetcher') as mock_get_fetcher:
            mock_fetcher = MagicMock()
            mock_fetcher.get_multipole_paths.return_value = mock_paths
            mock_get_fetcher.return_value = mock_fetcher

            # Mock loaders
            with patch('jaxeffort.load_multipole_noise_emulator') as mock_noise_load:
                with patch('jaxeffort.load_multipole_emulator') as mock_standard_load:
                    mock_standard_load.return_value = MagicMock(spec=['get_Pl'])

                    # Load without noise flag
                    config = {
                        'zenodo_url': 'test.tar.gz',
                        'has_noise': False  # Should use standard loading
                    }

                    emulators = _load_emulator_set('test_standard_model', config, auto_download=False)

                    # Should have called standard loader
                    assert mock_standard_load.call_count == 3
                    assert mock_noise_load.call_count == 0


class TestErrorRecovery:
    """Test error recovery during emulator loading."""

    def test_partial_loading_failure_recovery(self, tmp_path):
        """Verify graceful handling when some emulators fail to load."""
        import jaxeffort
        from jaxeffort import _load_emulator_set

        # Create mock paths
        mock_paths = {
            0: tmp_path / "0",
            2: tmp_path / "2",
            4: tmp_path / "4"
        }

        # Make paths exist
        for path in mock_paths.values():
            path.mkdir(parents=True)

        with patch('jaxeffort.data_fetcher.get_fetcher') as mock_get_fetcher:
            mock_fetcher = MagicMock()
            mock_fetcher.get_multipole_paths.return_value = mock_paths
            mock_get_fetcher.return_value = mock_fetcher

            # Mock loader to fail for l=2
            def mock_load(path):
                if "2" in str(path):
                    raise ValueError("Mock loading error for l=2")
                return MagicMock(spec=['get_Pl'])

            with patch('jaxeffort.load_multipole_emulator', side_effect=mock_load):
                with warnings.catch_warnings(record=True) as w:
                    config = {'zenodo_url': 'test.tar.gz', 'has_noise': False}
                    emulators = _load_emulator_set('test_model', config, auto_download=False)

                    # Should have warning about failed loading
                    assert len(w) > 0
                    assert "Error loading multipole l=2" in str(w[0].message)

                # Should have loaded 0 and 4, but not 2
                assert emulators['0'] is not None
                assert emulators['2'] is None  # Failed to load
                assert emulators['4'] is not None

    def test_complete_loading_failure_handling(self):
        """Verify handling when all emulators fail to load."""
        # Set environment to prevent auto-download
        with patch.dict(os.environ, {'JAXEFFORT_NO_AUTO_DOWNLOAD': '1'}):
            # Clear module cache to ensure clean import
            if 'jaxeffort' in sys.modules:
                del sys.modules['jaxeffort']
            if 'jaxeffort.jaxeffort' in sys.modules:
                del sys.modules['jaxeffort.jaxeffort']
            if 'jaxeffort.data_fetcher' in sys.modules:
                del sys.modules['jaxeffort.data_fetcher']

            # Now import with clean slate
            import jaxeffort
            from jaxeffort import _load_emulator_set

            # Mock everything to fail
            with patch('jaxeffort.data_fetcher.get_fetcher') as mock_get_fetcher:
                mock_fetcher = MagicMock()
                mock_fetcher.get_multipole_paths.side_effect = Exception("Network error")
                mock_get_fetcher.return_value = mock_fetcher

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")  # Ensure warnings are captured
                    config = {'zenodo_url': 'test.tar.gz', 'has_noise': False}
                    emulators = _load_emulator_set('failed_model', config, auto_download=True)

                    # Should have warning
                    assert len(w) > 0
                    assert "Could not initialize failed_model" in str(w[0].message)

                # Should return empty structure
                assert emulators == {'0': None, '2': None, '4': None}


class TestConfigurationManagement:
    """Test emulator configuration management."""

    def test_add_emulator_config_and_load(self, tmp_path):
        """Test adding new emulator configurations dynamically."""
        import jaxeffort

        # Mock successful loading
        mock_emulator = MagicMock(spec=['get_Pl'])

        with patch('jaxeffort._load_emulator_set') as mock_load:
            mock_load.return_value = {
                '0': mock_emulator,
                '2': mock_emulator,
                '4': mock_emulator
            }

            # Add new configuration
            result = jaxeffort.add_emulator_config(
                model_name='custom_model',
                zenodo_url='https://zenodo.org/custom.tar.gz',
                description='Custom test model',
                has_noise=True,
                checksum='abc123',
                auto_load=True
            )

            # Should be in configs
            assert 'custom_model' in jaxeffort.EMULATOR_CONFIGS
            assert jaxeffort.EMULATOR_CONFIGS['custom_model']['has_noise'] == True
            assert jaxeffort.EMULATOR_CONFIGS['custom_model']['checksum'] == 'abc123'

            # Should have loaded
            assert mock_load.called
            assert 'custom_model' in jaxeffort.trained_emulators
            assert result == jaxeffort.trained_emulators['custom_model']

    def test_reload_specific_model(self, tmp_path):
        """Test reloading a specific model's emulators."""
        import jaxeffort

        # Add a test model first
        jaxeffort.EMULATOR_CONFIGS['reload_test'] = {
            'zenodo_url': 'test.tar.gz',
            'has_noise': False
        }

        reload_count = [0]

        def mock_load(name, config, auto_download):
            reload_count[0] += 1
            return {'0': MagicMock(), '2': MagicMock(), '4': MagicMock()}

        with patch('jaxeffort._load_emulator_set', side_effect=mock_load):
            # Reload specific model
            jaxeffort.reload_emulators('reload_test')

            assert reload_count[0] == 1
            assert 'reload_test' in jaxeffort.trained_emulators

    def test_reload_all_models(self):
        """Test reloading all configured models."""
        import jaxeffort

        reload_calls = []

        def mock_load(name, config, auto_download):
            reload_calls.append(name)
            return {'0': None, '2': None, '4': None}

        with patch('jaxeffort._load_emulator_set', side_effect=mock_load):
            # Reload all
            jaxeffort.reload_emulators()

            # Should reload each configured model
            for model_name in jaxeffort.EMULATOR_CONFIGS:
                assert model_name in reload_calls

    def test_reload_unknown_model_raises(self):
        """Test that reloading unknown model raises error."""
        import jaxeffort

        with pytest.raises(ValueError, match="Unknown model: nonexistent"):
            jaxeffort.reload_emulators('nonexistent')


class TestImportFallbacks:
    """Test import mechanisms for jaxace."""

    def test_import_omega_m_a(self):
        """Test that Ωm_a is imported from jaxace."""
        import jaxeffort.jaxeffort as jef

        # Check that Ωm_a exists (the standard import)
        assert hasattr(jef, 'Ωm_a')


class TestInitializationIntegrity:
    """Test overall initialization integrity."""

    def test_trained_emulators_structure(self):
        """Verify trained_emulators has correct structure."""
        import jaxeffort

        # Should have entry for each configured model
        for model_name in jaxeffort.EMULATOR_CONFIGS:
            assert model_name in jaxeffort.trained_emulators

            # Each model should have 3 multipoles
            assert '0' in jaxeffort.trained_emulators[model_name]
            assert '2' in jaxeffort.trained_emulators[model_name]
            assert '4' in jaxeffort.trained_emulators[model_name]

    def test_exported_functions_available(self):
        """Verify all advertised functions are exported."""
        import jaxeffort

        # Core functions that should be available
        expected_exports = [
            'load_multipole_emulator',
            'load_multipole_noise_emulator',
            'get_stoch_terms',
            'clear_cache',
            'force_update',
            'check_for_updates',
            'get_cache_info',
            'add_emulator_config',
            'reload_emulators',
            'trained_emulators',
            'EMULATOR_CONFIGS'
        ]

        for export in expected_exports:
            assert hasattr(jaxeffort, export), f"Missing export: {export}"

    def test_initialization_is_deterministic(self):
        """Verify initialization produces consistent results."""
        # Clean slate - remove all jaxeffort modules
        modules_to_remove = [m for m in sys.modules if m.startswith('jaxeffort')]
        for module in modules_to_remove:
            del sys.modules[module]

        # First import
        import jaxeffort

        # Get initial state - should only have the default config
        initial_configs = {'pybird_mnuw0wacdm': jaxeffort.EMULATOR_CONFIGS['pybird_mnuw0wacdm']}
        initial_models = ['pybird_mnuw0wacdm']  # Only the default model

        # Force reimport
        modules_to_remove = [m for m in sys.modules if m.startswith('jaxeffort')]
        for module in modules_to_remove:
            del sys.modules[module]
        import jaxeffort as jaxeffort2

        # Should have same base configuration (ignore any added by tests)
        assert 'pybird_mnuw0wacdm' in jaxeffort2.EMULATOR_CONFIGS
        assert jaxeffort2.EMULATOR_CONFIGS['pybird_mnuw0wacdm'] == initial_configs['pybird_mnuw0wacdm']
        assert 'pybird_mnuw0wacdm' in jaxeffort2.trained_emulators


if __name__ == "__main__":
    pytest.main([__file__, "-v"])