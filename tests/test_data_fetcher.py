"""
Tests for the data fetcher module.

This module tests the data downloading, caching, and management
functionality of the MultipoleDataFetcher class.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
import jax.numpy as jnp


class TestDataFetcherErrorPaths:
    """Test data fetcher error handling."""

    def test_clear_cache(self):
        """Test clearing the cache."""
        from jaxeffort.data_fetcher import MultipoleDataFetcher

        # Create a fetcher with a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = MultipoleDataFetcher(
                zenodo_url="https://example.com/test.tar.gz",
                emulator_name="test",
                cache_dir=tmpdir
            )

            # Create some dummy files to clear
            emulator_dir = Path(tmpdir) / "emulators" / "test"
            emulator_dir.mkdir(parents=True, exist_ok=True)
            (emulator_dir / "test.txt").write_text("test")

            tar_path = Path(tmpdir) / "test.tar.gz"
            tar_path.write_text("dummy tar")

            # Clear cache
            fetcher.clear_cache()

            # Check files are gone
            assert not emulator_dir.exists()
            assert not tar_path.exists()

    def test_download_failure(self):
        """Test handling of download failures."""
        from jaxeffort.data_fetcher import MultipoleDataFetcher

        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = MultipoleDataFetcher(
                zenodo_url="https://nonexistent.example.com/nonexistent.tar.gz",
                emulator_name="test",
                cache_dir=tmpdir
            )

            # This should return False on failure
            success = fetcher.download_and_extract(show_progress=False)
            assert not success

    def test_checksum_verification_failure(self):
        """Test checksum verification failure."""
        from jaxeffort.data_fetcher import MultipoleDataFetcher

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy tar file
            tar_path = Path(tmpdir) / "test.tar.gz"
            tar_path.write_bytes(b"dummy content")

            fetcher = MultipoleDataFetcher(
                zenodo_url="file://" + str(tar_path),
                emulator_name="test",
                cache_dir=tmpdir,
                expected_checksum="wrong_checksum"
            )

            # Mock the download to succeed but checksum to fail
            with patch.object(fetcher, '_download_file', return_value=True):
                with patch.object(fetcher, '_verify_checksum', return_value=False):
                    success = fetcher.download_and_extract(show_progress=False)
                    assert not success

    def test_extraction_failure(self):
        """Test handling of extraction failures."""
        from jaxeffort.data_fetcher import MultipoleDataFetcher

        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = MultipoleDataFetcher(
                zenodo_url="https://example.com/test.tar.gz",
                emulator_name="test",
                cache_dir=tmpdir
            )

            # Create a corrupt tar file
            tar_path = Path(tmpdir) / "test.tar.gz"
            tar_path.write_bytes(b"not a valid tar file")

            # Mock download to succeed but extraction to fail
            with patch.object(fetcher, '_download_file', return_value=True):
                # Force tar_path to exist
                fetcher.tar_path = tar_path
                success = fetcher.download_and_extract(show_progress=False)
                assert not success


class TestConvenienceFunctions:
    """Test convenience functions in data_fetcher."""

    def test_get_emulator_path(self):
        """Test get_emulator_path convenience function."""
        from jaxeffort.data_fetcher import get_emulator_path

        # This should return the path to the cached emulator
        path = get_emulator_path()
        if path is not None:
            assert path.exists()
            assert (path / "0").exists()  # Check for multipole folder

    def test_get_multipole_paths(self):
        """Test get_multipole_paths convenience function."""
        from jaxeffort.data_fetcher import get_multipole_paths

        # This should return paths to individual multipoles
        paths = get_multipole_paths()
        if paths is not None:
            assert 0 in paths
            assert 2 in paths
            assert 4 in paths
            assert paths[0].exists()
            assert paths[2].exists()
            assert paths[4].exists()

    def test_get_fetcher_singleton(self):
        """Test that get_fetcher returns the same instance."""
        from jaxeffort.data_fetcher import get_fetcher, _default_fetcher

        # Get fetcher twice
        fetcher1 = get_fetcher()
        fetcher2 = get_fetcher()

        # Should be the same instance
        assert fetcher1 is fetcher2
        assert fetcher1 is _default_fetcher


class TestMultipoleFolderVerification:
    """Test multipole folder structure verification."""

    def test_verify_multipole_structure_complete(self, tmp_path):
        """Test verification of complete multipole structure."""
        from jaxeffort.data_fetcher import MultipoleDataFetcher

        # Create complete multipole structure
        for l in ["0", "2", "4"]:
            mp_dir = tmp_path / l
            mp_dir.mkdir()
            for comp in ["11", "loop", "ct"]:
                (mp_dir / comp).mkdir()

        fetcher = MultipoleDataFetcher(
            zenodo_url="https://example.com/test.tar.gz",
            emulator_name="test",
            cache_dir=tmp_path.parent
        )

        # Should verify successfully
        assert fetcher._verify_multipole_structure(tmp_path, show_progress=False)

    def test_verify_multipole_structure_incomplete(self, tmp_path):
        """Test verification of incomplete multipole structure."""
        from jaxeffort.data_fetcher import MultipoleDataFetcher

        # Create incomplete structure (missing l=4)
        for l in ["0", "2"]:
            mp_dir = tmp_path / l
            mp_dir.mkdir()
            for comp in ["11", "loop", "ct"]:
                (mp_dir / comp).mkdir()

        fetcher = MultipoleDataFetcher(
            zenodo_url="https://example.com/test.tar.gz",
            emulator_name="test",
            cache_dir=tmp_path.parent
        )

        # Should verify successfully (partial multipoles are OK)
        assert fetcher._verify_multipole_structure(tmp_path, show_progress=False)

    def test_verify_component_structure(self, tmp_path):
        """Test verification of component-only structure."""
        from jaxeffort.data_fetcher import MultipoleDataFetcher

        # Create component structure (no multipole folders)
        for comp in ["11", "loop", "ct"]:
            (tmp_path / comp).mkdir()

        fetcher = MultipoleDataFetcher(
            zenodo_url="https://example.com/test.tar.gz",
            emulator_name="test",
            cache_dir=tmp_path.parent
        )

        # Should verify successfully
        assert fetcher._verify_multipole_structure(tmp_path, show_progress=False)