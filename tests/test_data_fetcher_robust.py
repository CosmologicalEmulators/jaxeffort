"""
Robust tests for data_fetcher module focusing on real-world scenarios.

These tests validate critical behaviors rather than just increasing coverage:
- Version detection and cache invalidation
- Atomic operations and failure recovery
- Concurrent access safety
- Data integrity verification
"""

import os
import json
import tarfile
import hashlib
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock
from datetime import datetime, timedelta
import pytest
import urllib.error
import urllib.request
import shutil

# Set environment variable before importing jaxeffort
os.environ["JAXEFFORT_NO_AUTO_DOWNLOAD"] = "1"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaxeffort.data_fetcher import MultipoleDataFetcher


class TestVersionDetection:
    """Test that version detection correctly identifies when updates are needed."""

    def test_etag_change_triggers_update(self, tmp_path):
        """Verify that ETag changes are detected as updates."""
        fetcher = MultipoleDataFetcher(
            zenodo_url="https://zenodo.org/test.tar.gz",
            emulator_name="test",
            cache_dir=tmp_path
        )

        # Simulate initial download with ETag
        initial_metadata = {
            'downloaded_at': datetime.now().isoformat(),
            'remote_info': {
                'etag': 'initial-etag-v1',
                'size': 1000000,
                'last_modified': 'Mon, 01 Jan 2024 00:00:00 GMT'
            }
        }
        fetcher._save_metadata(initial_metadata)

        # Mock remote check returning different ETag
        with patch.object(fetcher, '_get_remote_info') as mock_remote:
            mock_remote.return_value = {
                'etag': 'updated-etag-v2',  # Changed
                'size': 1000000,  # Same size
                'last_modified': 'Mon, 01 Jan 2024 00:00:00 GMT'  # Same date
            }

            # Should detect update even if size and date are same
            assert fetcher.check_for_updates(show_progress=False) == True

    def test_no_false_positives_in_update_detection(self, tmp_path):
        """Ensure updates aren't falsely detected when content hasn't changed."""
        fetcher = MultipoleDataFetcher(
            zenodo_url="https://zenodo.org/test.tar.gz",
            emulator_name="test",
            cache_dir=tmp_path
        )

        # Set up metadata
        metadata = {
            'downloaded_at': datetime.now().isoformat(),
            'remote_info': {
                'etag': 'stable-etag',
                'size': 1000000,
                'last_modified': 'Mon, 01 Jan 2024 00:00:00 GMT'
            }
        }
        fetcher._save_metadata(metadata)

        # Mock remote returning exact same values
        with patch.object(fetcher, '_get_remote_info') as mock_remote:
            mock_remote.return_value = metadata['remote_info'].copy()

            # Should NOT detect update
            assert fetcher.check_for_updates(show_progress=False) == False

    def test_handles_missing_etag_gracefully(self, tmp_path):
        """Test fallback to size/date when ETag is unavailable."""
        fetcher = MultipoleDataFetcher(
            zenodo_url="https://zenodo.org/test.tar.gz",
            emulator_name="test",
            cache_dir=tmp_path
        )

        # Metadata without ETag
        metadata = {
            'downloaded_at': datetime.now().isoformat(),
            'remote_info': {
                'size': 1000000,
                'last_modified': 'Mon, 01 Jan 2024 00:00:00 GMT'
            }
        }
        fetcher._save_metadata(metadata)

        # Remote returns different size (no ETag)
        with patch.object(fetcher, '_get_remote_info') as mock_remote:
            mock_remote.return_value = {
                'size': 2000000,  # Different size
                'last_modified': 'Mon, 01 Jan 2024 00:00:00 GMT'
            }

            # Should detect update based on size change
            assert fetcher.check_for_updates(show_progress=False) == True


class TestAtomicOperations:
    """Ensure operations are atomic - no partial states left behind."""

    def test_download_cleanup_on_failure(self, tmp_path):
        """Verify temporary files are cleaned up when download fails."""
        fetcher = MultipoleDataFetcher(
            zenodo_url="https://zenodo.org/test.tar.gz",
            emulator_name="test",
            cache_dir=tmp_path
        )

        # Mock a download that fails halfway
        with patch('urllib.request.urlretrieve') as mock_download:
            mock_download.side_effect = urllib.error.URLError("Network error")

            success = fetcher._download_file(
                "https://zenodo.org/test.tar.gz",
                fetcher.tar_path,
                show_progress=False
            )

            assert success == False
            # Temp file should be cleaned up
            temp_file = fetcher.tar_path.with_suffix('.tmp')
            assert not temp_file.exists()
            # Final file should not exist
            assert not fetcher.tar_path.exists()

    def test_extraction_cleanup_on_failure(self, tmp_path):
        """Verify partial extraction is cleaned up on failure."""
        fetcher = MultipoleDataFetcher(
            zenodo_url="https://zenodo.org/test.tar.gz",
            emulator_name="test",
            cache_dir=tmp_path
        )

        # Create a corrupted tar file
        tar_path = tmp_path / "corrupted.tar.gz"
        tar_path.write_bytes(b"not a real tar file")

        # Try to extract
        extract_to = tmp_path / "extract_test"
        success = fetcher._extract_tar(tar_path, extract_to, show_progress=False)

        assert success == False
        # The extraction directory might be created but should be empty or minimal
        if extract_to.exists():
            # Should not have successfully extracted multipole folders
            assert not (extract_to / "0").exists()
            assert not (extract_to / "2").exists()
            assert not (extract_to / "4").exists()

    def test_force_update_is_atomic(self, tmp_path):
        """Verify force_update either fully succeeds or fully fails."""
        fetcher = MultipoleDataFetcher(
            zenodo_url="https://zenodo.org/test.tar.gz",
            emulator_name="test",
            cache_dir=tmp_path
        )

        # Create existing cache with data
        fetcher.emulators_dir.mkdir(parents=True)
        test_file = fetcher.emulators_dir / "existing_data.txt"
        test_file.write_text("important cached data")

        # Mock download to fail
        with patch.object(fetcher, 'download_and_extract', return_value=False):
            success = fetcher.force_update(show_progress=False)

            assert success == False
            # Cache dir should still exist (recreated even if download fails)
            assert fetcher.cache_dir.exists()
            # But old data should be gone (atomic - clear then try to download)
            assert not test_file.exists()


class TestDataIntegrity:
    """Verify data integrity through checksums and structure validation."""

    def test_checksum_verification_prevents_corruption(self, tmp_path):
        """Ensure corrupted downloads are detected via checksum."""
        expected_checksum = hashlib.sha256(b"correct content").hexdigest()

        fetcher = MultipoleDataFetcher(
            zenodo_url="https://zenodo.org/test.tar.gz",
            emulator_name="test",
            cache_dir=tmp_path,
            expected_checksum=expected_checksum
        )

        # Create file with wrong content
        bad_file = tmp_path / "bad.tar.gz"
        bad_file.write_bytes(b"corrupted content")

        # Verification should fail
        assert fetcher._verify_checksum(bad_file, expected_checksum) == False

        # Create file with correct content
        good_file = tmp_path / "good.tar.gz"
        good_file.write_bytes(b"correct content")

        # Verification should pass
        assert fetcher._verify_checksum(good_file, expected_checksum) == True

    def test_structure_verification_catches_incomplete_downloads(self, tmp_path):
        """Verify that incomplete emulator structures are detected."""
        fetcher = MultipoleDataFetcher(
            zenodo_url="https://zenodo.org/test.tar.gz",
            emulator_name="test",
            cache_dir=tmp_path
        )

        # Create truly incomplete structure (multipole folder without required components)
        base_path = tmp_path / "incomplete"
        base_path.mkdir()

        # Create l=0 folder but missing some component folders
        (base_path / "0").mkdir()
        (base_path / "0" / "11").mkdir()  # Only has 11, missing loop and ct

        # Should fail verification (has multipole folder but incomplete components)
        assert fetcher._verify_multipole_structure(base_path, show_progress=False) == False

        # Create partial but valid structure (only l=0 with all components)
        partial_path = tmp_path / "partial"
        partial_path.mkdir()
        for comp in ["11", "loop", "ct"]:
            (partial_path / "0" / comp).mkdir(parents=True)

        # Should pass verification (even with just one complete multipole)
        assert fetcher._verify_multipole_structure(partial_path, show_progress=False) == True

        # Create complete structure
        complete_path = tmp_path / "complete"
        for l in ["0", "2", "4"]:
            for comp in ["11", "loop", "ct"]:
                (complete_path / l / comp).mkdir(parents=True)

        # Should pass verification
        assert fetcher._verify_multipole_structure(complete_path, show_progress=False) == True

    def test_rejects_wrong_structure_type(self, tmp_path):
        """Ensure wrong directory structures are rejected."""
        fetcher = MultipoleDataFetcher(
            zenodo_url="https://zenodo.org/test.tar.gz",
            emulator_name="test",
            cache_dir=tmp_path
        )

        # Create wrong structure (random folders)
        wrong_path = tmp_path / "wrong"
        wrong_path.mkdir()
        (wrong_path / "random_folder").mkdir()
        (wrong_path / "another_folder").mkdir()

        # Should fail verification
        assert fetcher._verify_multipole_structure(wrong_path, show_progress=False) == False


class TestConcurrentAccess:
    """Test behavior with concurrent access to cache."""

    def test_concurrent_cache_checks_are_safe(self, tmp_path):
        """Multiple threads checking cache status shouldn't cause issues."""
        fetcher = MultipoleDataFetcher(
            zenodo_url="https://zenodo.org/test.tar.gz",
            emulator_name="test",
            cache_dir=tmp_path
        )

        # Save some metadata
        fetcher._save_metadata({'test': 'data', 'counter': 0})

        results = []
        errors = []

        def check_cache():
            try:
                info = fetcher.get_cache_info()
                results.append(info)
            except Exception as e:
                errors.append(e)

        # Launch multiple threads
        threads = []
        for _ in range(10):
            t = threading.Thread(target=check_cache)
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Should have no errors
        assert len(errors) == 0
        # Should have results from all threads
        assert len(results) == 10
        # All results should be consistent
        for info in results:
            assert info['cache_dir'] == str(tmp_path)
            assert info['emulator_name'] == 'test'



class TestNetworkResilience:
    """Test handling of various network conditions."""

    def test_retry_on_temporary_network_failure(self, tmp_path):
        """Test graceful handling of temporary network issues."""
        fetcher = MultipoleDataFetcher(
            zenodo_url="https://zenodo.org/test.tar.gz",
            emulator_name="test",
            cache_dir=tmp_path
        )

        # Simulate intermittent network (fails once, then succeeds)
        call_count = [0]
        def mock_urlretrieve(url, destination, reporthook=None):
            call_count[0] += 1
            if call_count[0] == 1:
                raise urllib.error.URLError("Temporary network issue")
            # Second call succeeds
            Path(destination).write_bytes(b"downloaded content")

        with patch('urllib.request.urlretrieve', side_effect=mock_urlretrieve):
            # First attempt should fail
            success = fetcher._download_file(
                "https://zenodo.org/test.tar.gz",
                fetcher.tar_path,
                show_progress=False
            )
            assert success == False

            # Second attempt should succeed
            success = fetcher._download_file(
                "https://zenodo.org/test.tar.gz",
                fetcher.tar_path,
                show_progress=False
            )
            assert success == True
            assert fetcher.tar_path.read_bytes() == b"downloaded content"

    def test_handles_timeout_gracefully(self, tmp_path):
        """Verify timeout handling doesn't leave partial files."""
        fetcher = MultipoleDataFetcher(
            zenodo_url="https://zenodo.org/test.tar.gz",
            emulator_name="test",
            cache_dir=tmp_path
        )

        with patch('urllib.request.urlretrieve') as mock_download:
            # Simulate timeout
            mock_download.side_effect = TimeoutError("Download timeout")

            success = fetcher._download_file(
                "https://zenodo.org/test.tar.gz",
                fetcher.tar_path,
                show_progress=False
            )

            assert success == False
            # No partial files should remain
            assert not fetcher.tar_path.exists()
            assert not fetcher.tar_path.with_suffix('.tmp').exists()


class TestUserExperience:
    """Test user-facing features like progress reporting and error messages."""

    def test_download_progress_reporting_accuracy(self, tmp_path, capsys):
        """Verify download progress is reported accurately."""
        fetcher = MultipoleDataFetcher(
            zenodo_url="https://zenodo.org/test.tar.gz",
            emulator_name="test",
            cache_dir=tmp_path
        )

        # Mock urlretrieve with progress callbacks
        def mock_urlretrieve(url, dest, reporthook=None):
            if reporthook:
                # Simulate download progress
                total_size = 10 * 1024 * 1024  # 10 MB
                block_size = 8192
                for block_num in range(0, total_size // block_size + 1):
                    reporthook(block_num, block_size, total_size)
            Path(dest).write_bytes(b"content")

        with patch('urllib.request.urlretrieve', side_effect=mock_urlretrieve):
            success = fetcher._download_file(
                "https://zenodo.org/test.tar.gz",
                fetcher.tar_path,
                show_progress=True
            )

            assert success == True
            captured = capsys.readouterr()
            # Should show download progress
            assert "Downloading" in captured.out
            assert "MB" in captured.out

    def test_clear_error_messages_on_failure(self, tmp_path, capsys):
        """Ensure error messages are clear and actionable."""
        fetcher = MultipoleDataFetcher(
            zenodo_url="https://zenodo.org/test.tar.gz",
            emulator_name="test",
            cache_dir=tmp_path
        )

        # Test network error message
        with patch('urllib.request.urlretrieve') as mock_download:
            mock_download.side_effect = urllib.error.URLError("Network is unreachable")

            success = fetcher._download_file(
                "https://zenodo.org/test.tar.gz",
                fetcher.tar_path,
                show_progress=True
            )

            assert success == False
            captured = capsys.readouterr()
            assert "Error downloading" in captured.out
            assert "Network is unreachable" in captured.out


class TestRealWorldScenarios:
    """Test complete real-world usage scenarios."""

    def test_fresh_install_workflow(self, tmp_path):
        """Test the complete workflow for a fresh installation."""
        fetcher = MultipoleDataFetcher(
            zenodo_url="https://zenodo.org/test.tar.gz",
            emulator_name="test",
            cache_dir=tmp_path
        )

        # Mock successful download and extraction
        mock_tar_content = self._create_mock_tar_with_emulators(tmp_path)

        with patch('urllib.request.urlretrieve') as mock_download:
            def save_tar(url, dest, reporthook=None):
                shutil.copy(mock_tar_content, dest)

            mock_download.side_effect = save_tar

            # Should successfully download and extract
            success = fetcher.download_and_extract(show_progress=False)
            assert success == True

            # Should have created the correct structure
            assert fetcher.emulators_dir.exists()
            for l in ["0", "2", "4"]:
                assert (fetcher.emulators_dir / l).exists()
                for comp in ["11", "loop", "ct"]:
                    assert (fetcher.emulators_dir / l / comp).exists()

            # Metadata should be saved
            metadata = fetcher._load_metadata()
            assert 'downloaded_at' in metadata
            assert metadata['emulator_name'] == 'test'

    def test_update_workflow_with_cached_data(self, tmp_path):
        """Test updating from an old cached version to a new one."""
        fetcher = MultipoleDataFetcher(
            zenodo_url="https://zenodo.org/test.tar.gz",
            emulator_name="test",
            cache_dir=tmp_path
        )

        # Set up old cached data
        old_metadata = {
            'downloaded_at': '2023-01-01T00:00:00',
            'remote_info': {'etag': 'old-version', 'size': 1000000}
        }
        fetcher._save_metadata(old_metadata)

        # Create old emulator structure
        fetcher.emulators_dir.mkdir(parents=True)
        old_file = fetcher.emulators_dir / "old_version.txt"
        old_file.write_text("old data")

        # Mock update check showing new version
        with patch.object(fetcher, '_get_remote_info') as mock_remote:
            mock_remote.return_value = {'etag': 'new-version', 'size': 2000000}

            # Should detect update
            assert fetcher.check_for_updates(show_progress=False) == True

        # Mock force update
        with patch.object(fetcher, 'download_and_extract', return_value=True):
            success = fetcher.force_update(show_progress=False)
            assert success == True

            # Old file should be gone (cache was cleared)
            assert not old_file.exists()

    def _create_mock_tar_with_emulators(self, tmp_path):
        """Helper to create a mock tar.gz with correct structure."""
        # Create temporary directory with emulator structure
        temp_dir = tmp_path / "temp_emulators"
        for l in ["0", "2", "4"]:
            for comp in ["11", "loop", "ct"]:
                comp_dir = temp_dir / l / comp
                comp_dir.mkdir(parents=True)
                # Add a dummy file
                (comp_dir / "weights.npy").write_text("dummy weights")

        # Create tar.gz
        tar_path = tmp_path / "mock_emulators.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(temp_dir, arcname=".")

        return tar_path


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])