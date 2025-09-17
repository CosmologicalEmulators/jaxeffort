"""
Tests for jaxeffort initialization and configuration.

This module tests the initialization behavior, configuration management,
and dynamic loading features of jaxeffort.
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch
import jax.numpy as jnp


class TestNoAutoDownload:
    """Test behavior when auto-download is disabled."""

    def test_no_auto_download_mode(self):
        """Test that emulators aren't downloaded when JAXEFFORT_NO_AUTO_DOWNLOAD is set."""
        # Save current environment
        old_env = os.environ.get('JAXEFFORT_NO_AUTO_DOWNLOAD')

        try:
            # Set no auto-download
            os.environ['JAXEFFORT_NO_AUTO_DOWNLOAD'] = '1'

            # Re-import jaxeffort to trigger the import-time logic
            import importlib
            import jaxeffort
            importlib.reload(jaxeffort)

            # Check that emulators are marked as disabled
            for model_name in jaxeffort.EMULATOR_CONFIGS:
                assert model_name in jaxeffort.trained_emulators
                assert not jaxeffort.trained_emulators[model_name].get("loaded", False)
                assert jaxeffort.trained_emulators[model_name].get("disabled", False)

        finally:
            # Restore environment
            if old_env is None:
                os.environ.pop('JAXEFFORT_NO_AUTO_DOWNLOAD', None)
            else:
                os.environ['JAXEFFORT_NO_AUTO_DOWNLOAD'] = old_env

            # Reload to restore normal behavior
            import importlib
            import jaxeffort
            importlib.reload(jaxeffort)


class TestDynamicEmulatorConfig:
    """Test dynamic emulator configuration functions."""

    def test_add_emulator_config(self):
        """Test adding a new emulator configuration."""
        import jaxeffort

        # Add a test configuration (without auto_load to avoid actual download)
        result = jaxeffort.add_emulator_config(
            model_name="test_model",
            zenodo_url="https://example.com/test.tar.gz",
            description="Test emulator",
            has_noise=True,
            checksum="abc123",
            auto_load=False
        )

        # Check it was added
        assert "test_model" in jaxeffort.EMULATOR_CONFIGS
        assert jaxeffort.EMULATOR_CONFIGS["test_model"]["zenodo_url"] == "https://example.com/test.tar.gz"
        assert jaxeffort.EMULATOR_CONFIGS["test_model"]["description"] == "Test emulator"
        assert jaxeffort.EMULATOR_CONFIGS["test_model"]["has_noise"] is True
        assert jaxeffort.EMULATOR_CONFIGS["test_model"]["checksum"] == "abc123"

        # Check result
        assert not result.get("loaded", False)

        # Clean up
        del jaxeffort.EMULATOR_CONFIGS["test_model"]
        del jaxeffort.trained_emulators["test_model"]

    def test_reload_emulators_specific_model(self):
        """Test reloading a specific emulator."""
        import jaxeffort

        # This should reload the pybird model (which is already loaded)
        result = jaxeffort.reload_emulators("pybird_mnuw0wacdm")

        assert "pybird_mnuw0wacdm" in result
        assert result["pybird_mnuw0wacdm"].get("loaded", False)

    def test_reload_emulators_unknown_model(self):
        """Test reloading an unknown model raises error."""
        import jaxeffort

        with pytest.raises(ValueError, match="Unknown model"):
            jaxeffort.reload_emulators("nonexistent_model")

    def test_reload_all_emulators(self):
        """Test reloading all emulators."""
        import jaxeffort

        # This should reload all configured models
        result = jaxeffort.reload_emulators()

        # Check all configured models are in result
        for model_name in jaxeffort.EMULATOR_CONFIGS:
            assert model_name in result


class TestEmulatorAccessFunctions:
    """Test emulator access functions and edge cases."""

    def test_get_default_emulator_with_multipole(self):
        """Test getting default emulator for specific multipole."""
        import jaxeffort

        # Get l=0 emulator
        emulator = jaxeffort.get_default_emulator(l=0)
        assert emulator is not None

        # Get l=2 emulator
        emulator = jaxeffort.get_default_emulator(l=2)
        assert emulator is not None

        # Get all multipoles
        emulators = jaxeffort.get_default_emulator(l=None)
        assert emulators is not None
        assert 0 in emulators
        assert 2 in emulators
        assert 4 in emulators

    def test_get_multipole_emulator_missing_model(self):
        """Test get_multipole_emulator with missing model."""
        import jaxeffort

        emulator = jaxeffort.get_multipole_emulator("nonexistent_model", l=0)
        assert emulator is None

    def test_get_multipole_emulator_missing_multipole(self):
        """Test get_multipole_emulator with invalid multipole."""
        import jaxeffort

        # Temporarily modify the multipoles to test edge case
        if "pybird_mnuw0wacdm" in jaxeffort.trained_emulators:
            emulator_data = jaxeffort.trained_emulators["pybird_mnuw0wacdm"]
            if emulator_data.get("loaded") and "multipoles" in emulator_data:
                # Try to get a non-existent multipole
                emulator = jaxeffort.get_multipole_emulator("pybird_mnuw0wacdm", l=6)
                assert emulator is None


class TestEmulatorLoadingErrors:
    """Test error handling during emulator loading."""

    def test_load_emulator_with_exception(self):
        """Test emulator loading with exceptions."""
        import jaxeffort
        from jaxeffort import _load_emulator_set

        # Create a config that will fail
        bad_config = {
            "zenodo_url": "https://nonexistent.example.com/bad.tar.gz",
            "description": "Bad emulator",
            "has_noise": False
        }

        # This should handle the exception gracefully - patch to simulate error
        with patch('jaxeffort.data_fetcher.MultipoleDataFetcher.get_multipole_paths', return_value=None):
            result = _load_emulator_set("bad_model", bad_config, auto_download=False)

            assert not result.get("loaded", False)