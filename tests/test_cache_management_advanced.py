"""
Advanced cache management tests to improve coverage.

This module tests the cache management functionality including:
- Metadata persistence
- Update checking
- Force updates
- Cache info retrieval
"""

import os
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime
import pytest
import urllib.error

# Set environment variable before importing jaxeffort
os.environ["JAXEFFORT_NO_AUTO_DOWNLOAD"] = "1"

# Import directly from data_fetcher module to avoid JAX initialization issues
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaxeffort.data_fetcher import (
    MultipoleDataFetcher,
    clear_cache,
    check_for_updates,
    force_update,
    get_cache_info,
    clear_all_cache
)


class TestMetadataHandling:
    """Test metadata loading and saving functionality."""

    @pytest.fixture
    def temp_cache(self, tmp_path):
        """Create a temporary cache directory."""
        cache_dir = tmp_path / ".jaxeffort_data"
        cache_dir.mkdir()
        return cache_dir

    def test_save_and_load_metadata(self, temp_cache):
        """Test that metadata can be saved and loaded correctly."""
        fetcher = MultipoleDataFetcher(
            zenodo_url="https://example.com/test.tar.gz",
            emulator_name="test_model",
            cache_dir=temp_cache
        )

        # Create test metadata
        test_metadata = {
            'downloaded_at': '2024-01-01T00:00:00',
            'zenodo_url': 'https://example.com/test.tar.gz',
            'remote_info': {
                'etag': 'test-etag-123',
                'size': 1234567,
                'last_modified': 'Mon, 01 Jan 2024 00:00:00 GMT'
            }
        }

        # Save metadata
        fetcher._save_metadata(test_metadata)

        # Verify file exists
        assert fetcher.metadata_file.exists()

        # Load metadata
        loaded = fetcher._load_metadata()

        # Verify content
        assert loaded == test_metadata
        assert loaded['remote_info']['etag'] == 'test-etag-123'

    def test_load_metadata_missing_file(self, temp_cache):
        """Test loading metadata when file doesn't exist."""
        fetcher = MultipoleDataFetcher(
            zenodo_url="https://example.com/test.tar.gz",
            emulator_name="test_model",
            cache_dir=temp_cache
        )

        # Ensure file doesn't exist
        if fetcher.metadata_file.exists():
            fetcher.metadata_file.unlink()

        # Should return empty dict
        metadata = fetcher._load_metadata()
        assert metadata == {}

    def test_load_metadata_corrupted_json(self, temp_cache):
        """Test loading metadata with corrupted JSON."""
        fetcher = MultipoleDataFetcher(
            zenodo_url="https://example.com/test.tar.gz",
            emulator_name="test_model",
            cache_dir=temp_cache
        )

        # Write corrupted JSON
        fetcher.metadata_file.write_text("not valid json {")

        # Should return empty dict
        metadata = fetcher._load_metadata()
        assert metadata == {}

    @patch('builtins.open', side_effect=IOError("Permission denied"))
    def test_save_metadata_permission_error(self, mock_open, temp_cache, capsys):
        """Test handling of permission errors when saving metadata."""
        fetcher = MultipoleDataFetcher(
            zenodo_url="https://example.com/test.tar.gz",
            emulator_name="test_model",
            cache_dir=temp_cache
        )

        # Try to save metadata
        fetcher._save_metadata({'test': 'data'})

        # Should print warning
        captured = capsys.readouterr()
        assert "Warning: Could not save metadata" in captured.out


class TestRemoteInfoRetrieval:
    """Test remote file information retrieval."""

    def test_get_remote_info_success(self):
        """Test successful retrieval of remote file info."""
        fetcher = MultipoleDataFetcher(
            zenodo_url="https://example.com/test.tar.gz",
            emulator_name="test_model"
        )

        # Mock response
        mock_response = MagicMock()
        mock_response.headers = {
            'Content-Length': '1234567',
            'Last-Modified': 'Mon, 01 Jan 2024 00:00:00 GMT',
            'ETag': '"test-etag-123"'
        }

        with patch('urllib.request.urlopen', return_value=mock_response):
            info = fetcher._get_remote_info("https://example.com/test.tar.gz")

        assert info['size'] == 1234567
        assert info['last_modified'] == 'Mon, 01 Jan 2024 00:00:00 GMT'
        assert info['etag'] == 'test-etag-123'

    def test_get_remote_info_network_error(self):
        """Test handling of network errors when getting remote info."""
        fetcher = MultipoleDataFetcher(
            zenodo_url="https://example.com/test.tar.gz",
            emulator_name="test_model"
        )

        with patch('urllib.request.urlopen', side_effect=urllib.error.URLError("Network error")):
            info = fetcher._get_remote_info("https://example.com/test.tar.gz")

        # Should return empty dict on error
        assert info == {}

    def test_get_remote_info_partial_headers(self):
        """Test handling of partial headers in response."""
        fetcher = MultipoleDataFetcher(
            zenodo_url="https://example.com/test.tar.gz",
            emulator_name="test_model"
        )

        # Mock response with only some headers
        mock_response = MagicMock()
        mock_response.headers = {
            'Content-Length': '1234567',
            # Missing Last-Modified and ETag
        }

        with patch('urllib.request.urlopen', return_value=mock_response):
            info = fetcher._get_remote_info("https://example.com/test.tar.gz")

        assert info['size'] == 1234567
        assert 'last_modified' not in info
        assert 'etag' not in info


class TestUpdateChecking:
    """Test update checking functionality."""

    @pytest.fixture
    def fetcher_with_metadata(self, tmp_path):
        """Create fetcher with existing metadata."""
        cache_dir = tmp_path / ".jaxeffort_data"
        cache_dir.mkdir()

        fetcher = MultipoleDataFetcher(
            zenodo_url="https://example.com/test.tar.gz",
            emulator_name="test_model",
            cache_dir=cache_dir
        )

        # Save initial metadata
        metadata = {
            'downloaded_at': '2024-01-01T00:00:00',
            'remote_info': {
                'etag': 'old-etag',
                'size': 1000000,
                'last_modified': 'Mon, 01 Jan 2024 00:00:00 GMT'
            }
        }
        fetcher._save_metadata(metadata)

        return fetcher

    def test_check_for_updates_etag_changed(self, fetcher_with_metadata, capsys):
        """Test update detection via ETag change."""
        # Mock remote info with different ETag
        with patch.object(fetcher_with_metadata, '_get_remote_info') as mock_remote:
            mock_remote.return_value = {
                'etag': 'new-etag',  # Different ETag
                'size': 1000000,
                'last_modified': 'Mon, 01 Jan 2024 00:00:00 GMT'
            }

            update_available = fetcher_with_metadata.check_for_updates()

        assert update_available == True
        captured = capsys.readouterr()
        assert "Update available: ETag changed" in captured.out

    def test_check_for_updates_size_changed(self, fetcher_with_metadata, capsys):
        """Test update detection via file size change."""
        # Mock remote info with different size
        with patch.object(fetcher_with_metadata, '_get_remote_info') as mock_remote:
            mock_remote.return_value = {
                'size': 2000000,  # Different size
                'last_modified': 'Mon, 01 Jan 2024 00:00:00 GMT'
            }

            update_available = fetcher_with_metadata.check_for_updates()

        assert update_available == True
        captured = capsys.readouterr()
        assert "Update available: File size changed" in captured.out

    def test_check_for_updates_no_changes(self, fetcher_with_metadata, capsys):
        """Test when no updates are available."""
        # Mock remote info with same values
        with patch.object(fetcher_with_metadata, '_get_remote_info') as mock_remote:
            mock_remote.return_value = {
                'etag': 'old-etag',  # Same ETag
                'size': 1000000,  # Same size
                'last_modified': 'Mon, 01 Jan 2024 00:00:00 GMT'
            }

            update_available = fetcher_with_metadata.check_for_updates()

        assert update_available == False
        captured = capsys.readouterr()
        assert "Cached version is up to date" in captured.out

    def test_check_for_updates_no_metadata(self, tmp_path, capsys):
        """Test update checking when no metadata exists."""
        cache_dir = tmp_path / ".jaxeffort_data"
        cache_dir.mkdir()

        fetcher = MultipoleDataFetcher(
            zenodo_url="https://example.com/test.tar.gz",
            emulator_name="test_model",
            cache_dir=cache_dir
        )

        # Ensure no metadata exists
        if fetcher.metadata_file.exists():
            fetcher.metadata_file.unlink()

        update_available = fetcher.check_for_updates()

        assert update_available == True
        captured = capsys.readouterr()
        assert "No cached metadata found" in captured.out


class TestCacheInfo:
    """Test cache information retrieval."""

    def test_get_cache_info_with_data(self, tmp_path):
        """Test cache info retrieval with cached data."""
        cache_dir = tmp_path / ".jaxeffort_data"
        cache_dir.mkdir()

        fetcher = MultipoleDataFetcher(
            zenodo_url="https://example.com/test.tar.gz",
            emulator_name="test_model",
            cache_dir=cache_dir
        )

        # Create mock emulator directory with files
        fetcher.emulators_dir.mkdir(parents=True)
        (fetcher.emulators_dir / "test_file.txt").write_text("test content")

        # Create tar file
        fetcher.tar_path.write_text("mock tar content")

        # Save metadata
        metadata = {
            'downloaded_at': '2024-01-01T00:00:00',
            'checksum_verified': True
        }
        fetcher._save_metadata(metadata)

        # Get cache info
        info = fetcher.get_cache_info()

        assert info['cache_dir'] == str(cache_dir)
        assert info['emulator_name'] == 'test_model'
        assert info['has_cached_data'] == True
        assert 'extracted_size_mb' in info
        assert 'tar_size_mb' in info
        assert info['downloaded_at'] == '2024-01-01T00:00:00'
        assert info['checksum_verified'] == True

    def test_get_cache_info_empty(self, tmp_path):
        """Test cache info when no data is cached."""
        cache_dir = tmp_path / ".jaxeffort_data"
        cache_dir.mkdir()

        fetcher = MultipoleDataFetcher(
            zenodo_url="https://example.com/test.tar.gz",
            emulator_name="test_model",
            cache_dir=cache_dir
        )

        info = fetcher.get_cache_info()

        assert info['cache_dir'] == str(cache_dir)
        assert info['emulator_name'] == 'test_model'
        assert info['has_cached_data'] == False
        assert 'extracted_size_mb' not in info
        assert 'tar_size_mb' not in info


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    @patch('jaxeffort.data_fetcher.get_fetcher')
    def test_clear_cache_convenience(self, mock_get_fetcher):
        """Test the module-level clear_cache function."""
        mock_fetcher = MagicMock()
        mock_get_fetcher.return_value = mock_fetcher

        # Test default model
        clear_cache(show_progress=False)
        mock_fetcher.clear_cache.assert_called_once_with(clear_tar=True, show_progress=False)

        # Test with specific model
        mock_fetcher.reset_mock()
        with patch('jaxeffort.data_fetcher.MultipoleDataFetcher') as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            clear_cache(model_name="test_model", clear_tar=False)
            mock_instance.clear_cache.assert_called_once_with(clear_tar=False, show_progress=True)

    @patch('jaxeffort.data_fetcher.get_fetcher')
    def test_check_for_updates_convenience(self, mock_get_fetcher):
        """Test the module-level check_for_updates function."""
        mock_fetcher = MagicMock()
        mock_fetcher.check_for_updates.return_value = True
        mock_get_fetcher.return_value = mock_fetcher

        result = check_for_updates(show_progress=False)

        assert result == True
        mock_fetcher.check_for_updates.assert_called_once_with(show_progress=False)

    @patch('jaxeffort.data_fetcher.Path')
    def test_clear_all_cache(self, mock_path, capsys):
        """Test clearing all cached data."""
        # Mock the path operations
        mock_cache_dir = MagicMock()
        mock_cache_dir.exists.return_value = True
        mock_path.home.return_value = MagicMock()
        mock_path.home.return_value.__truediv__.return_value = mock_cache_dir

        with patch('shutil.rmtree') as mock_rmtree:
            clear_all_cache()

            mock_rmtree.assert_called_once_with(mock_cache_dir)
            captured = capsys.readouterr()
            assert "Cleared all cached data" in captured.out

    @patch('jaxeffort.data_fetcher.Path')
    def test_clear_all_cache_no_data(self, mock_path, capsys):
        """Test clearing when no cache exists."""
        # Mock non-existent cache directory
        mock_cache_dir = MagicMock()
        mock_cache_dir.exists.return_value = False
        mock_path.home.return_value = MagicMock()
        mock_path.home.return_value.__truediv__.return_value = mock_cache_dir

        clear_all_cache()

        captured = capsys.readouterr()
        assert "No cached data to clear" in captured.out


class TestForceUpdate:
    """Test force update functionality."""

    def test_force_update_complete_cycle(self, tmp_path, capsys):
        """Test complete force update process."""
        cache_dir = tmp_path / ".jaxeffort_data"
        cache_dir.mkdir()

        # Create existing files
        old_file = cache_dir / "old_file.txt"
        old_file.write_text("old data")

        fetcher = MultipoleDataFetcher(
            zenodo_url="https://example.com/test.tar.gz",
            emulator_name="test_model",
            cache_dir=cache_dir
        )

        # Mock the download_and_extract method
        with patch.object(fetcher, 'download_and_extract', return_value=True):
            success = fetcher.force_update()

        assert success == True
        # Old file should be gone
        assert not old_file.exists()
        # Cache directory should be recreated
        assert cache_dir.exists()

        captured = capsys.readouterr()
        assert "Force updating multipole emulator data" in captured.out
        assert "Cleared cache directory" in captured.out
        assert "Recreated cache directory" in captured.out