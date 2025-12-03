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

            # Check that emulators have None values when disabled
            for model_name in jaxeffort.list_emulators():
                assert model_name in jaxeffort.trained_emulators
                # Should have keys "0", "2", "4" all set to None
                assert jaxeffort.trained_emulators[model_name]["0"] is None
                assert jaxeffort.trained_emulators[model_name]["2"] is None
                assert jaxeffort.trained_emulators[model_name]["4"] is None

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

    def test_list_emulators(self):
        """Test listing available emulators from Artifacts.toml."""
        import jaxeffort

        emulators = jaxeffort.list_emulators()

        # Should have at least the three default emulators
        assert "pybird_mnuw0wacdm" in emulators
        assert "velocileptors_lpt_mnuw0wacdm" in emulators
        assert "velocileptors_rept_mnuw0wacdm" in emulators

    def test_get_emulator_info(self):
        """Test getting emulator info from Artifacts.toml."""
        import jaxeffort

        info = jaxeffort.get_emulator_info("pybird_mnuw0wacdm")

        assert info["name"] == "pybird_mnuw0wacdm"
        assert "git_tree_sha1" in info
        assert "description" in info
        assert "has_noise" in info

    def test_reload_emulators_specific_model(self):
        """Test reloading a specific emulator."""
        import jaxeffort

        # Skip if emulators not available
        if jaxeffort.trained_emulators.get("pybird_mnuw0wacdm", {}).get("0") is None:
            pytest.skip("Emulators not downloaded")

        # This should reload the pybird model (which is already loaded)
        result = jaxeffort.reload_emulators("pybird_mnuw0wacdm")

        assert "pybird_mnuw0wacdm" in result
        # Check if at least one multipole is loaded (new structure)
        loaded = sum(1 for v in result["pybird_mnuw0wacdm"].values() if v is not None)
        assert loaded > 0

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
        for model_name in jaxeffort.list_emulators():
            assert model_name in result


class TestEmulatorAccessFunctions:
    """Test emulator access functions and edge cases."""

    def test_get_default_emulator_with_multipole(self):
        """Test accessing emulators through trained_emulators dict."""
        import jaxeffort

        # Skip if emulators not available
        if jaxeffort.trained_emulators.get("pybird_mnuw0wacdm", {}).get("0") is None:
            pytest.skip("Emulators not downloaded")

        # Get l=0 emulator via new simplified structure
        emulator_0 = jaxeffort.trained_emulators["pybird_mnuw0wacdm"]["0"]
        assert emulator_0 is not None

        # Get l=2 emulator
        emulator_2 = jaxeffort.trained_emulators["pybird_mnuw0wacdm"]["2"]
        assert emulator_2 is not None

        # Get l=4 emulator
        emulator_4 = jaxeffort.trained_emulators["pybird_mnuw0wacdm"]["4"]
        assert emulator_4 is not None

    def test_get_multipole_emulator_missing_model(self):
        """Test accessing non-existent model."""
        import jaxeffort

        # Try to access a model that doesn't exist
        assert "nonexistent_model" not in jaxeffort.trained_emulators

    def test_get_multipole_emulator_missing_multipole(self):
        """Test accessing invalid multipole."""
        import jaxeffort

        # The dictionary only has "0", "2", "4" keys
        # Accessing a non-existent key would raise KeyError
        assert "6" not in jaxeffort.trained_emulators["pybird_mnuw0wacdm"]


class TestEmulatorLoadingErrors:
    """Test error handling during emulator loading."""

    def test_load_emulator_with_nonexistent_model(self):
        """Test emulator loading with nonexistent model."""
        import jaxeffort
        from jaxeffort import _load_emulator_set

        # Loading a model that doesn't exist in Artifacts.toml should handle gracefully
        result = _load_emulator_set("nonexistent_model", auto_download=False)

        # Should return dict with None values
        assert result["0"] is None
        assert result["2"] is None
        assert result["4"] is None