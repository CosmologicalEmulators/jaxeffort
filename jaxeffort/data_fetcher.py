"""
Data fetcher for jaxeffort emulator files from Zenodo.

This module handles downloading, extracting, and caching of trained multipole emulator data.
Based on the jaxcapse data fetcher design but adapted for jaxeffort's multipole structure.
"""

import hashlib
import json
import os
import shutil
import tarfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
from urllib.error import URLError


class MultipoleDataFetcher:
    """
    Manages downloading and caching of multipole emulator data from Zenodo.

    The data is cached in ~/.jaxeffort_data/ by default.
    """

    def __init__(self,
                 zenodo_url: str,
                 emulator_name: str = "pybird_mnuw0wacdm",
                 cache_dir: Optional[Union[str, Path]] = None,
                 expected_checksum: Optional[str] = None):
        """
        Initialize the data fetcher.

        Parameters
        ----------
        zenodo_url : str
            URL to download the emulator tar.gz file from.
        emulator_name : str
            Name identifier for this emulator set.
        cache_dir : str or Path, optional
            Directory to cache downloaded files.
            Defaults to ~/.jaxeffort_data/
        expected_checksum : str, optional
            Expected SHA256 checksum of the downloaded file for verification.
        """
        # Store required parameters
        self.zenodo_url = zenodo_url
        self.emulator_name = emulator_name
        self.expected_checksum = expected_checksum

        if cache_dir is None:
            self.cache_dir = Path.home() / ".jaxeffort_data"
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Path for the downloaded tar.gz file
        # Extract filename from URL
        tar_filename = self.zenodo_url.split('/')[-1].split('?')[0]
        self.tar_path = self.cache_dir / tar_filename

        # Path for extracted emulators
        self.emulators_dir = self.cache_dir / "emulators" / self.emulator_name

        # Path for metadata file
        self.metadata_file = self.cache_dir / f"{self.emulator_name}_metadata.json"

    def _download_file(self, url: str, destination: Path,
                      show_progress: bool = True) -> bool:
        """
        Download a file from URL to destination.

        Parameters
        ----------
        url : str
            URL to download from
        destination : Path
            Local path to save the file
        show_progress : bool
            Whether to show download progress

        Returns
        -------
        bool
            True if download successful, False otherwise
        """
        try:
            # Create temporary file for download
            temp_file = destination.with_suffix('.tmp')

            def download_hook(block_num, block_size, total_size):
                if show_progress and total_size > 0:
                    downloaded = block_num * block_size
                    percent = min(downloaded * 100 / total_size, 100)
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(f"\rDownloading: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)",
                          end='', flush=True)

            if show_progress:
                print(f"Downloading multipole emulator data from Zenodo...")

            urllib.request.urlretrieve(url, temp_file,
                                      reporthook=download_hook if show_progress else None)

            if show_progress:
                print()  # New line after progress

            # Move temp file to final destination
            shutil.move(str(temp_file), str(destination))
            return True

        except (URLError, IOError) as e:
            if show_progress:
                print(f"\nError downloading: {e}")
            # Clean up temp file if exists
            temp_file = destination.with_suffix('.tmp')
            if temp_file.exists():
                temp_file.unlink()
            return False

    def _extract_tar(self, tar_path: Path, extract_to: Path,
                    show_progress: bool = True) -> bool:
        """
        Extract tar.gz file.

        Parameters
        ----------
        tar_path : Path
            Path to the tar.gz file
        extract_to : Path
            Directory to extract files to
        show_progress : bool
            Whether to show extraction progress

        Returns
        -------
        bool
            True if extraction successful, False otherwise
        """
        try:
            if show_progress:
                print(f"Extracting multipole emulator data...")

            extract_to.mkdir(parents=True, exist_ok=True)

            with tarfile.open(tar_path, 'r:gz') as tar:
                # Extract all files
                tar.extractall(extract_to)

            if show_progress:
                print("Extraction complete!")

            return True

        except (tarfile.TarError, IOError) as e:
            if show_progress:
                print(f"Error extracting tar file: {e}")
            return False

    def _load_metadata(self) -> Dict[str, Any]:
        """
        Load cached metadata about the downloaded files.

        Returns
        -------
        dict
            Metadata dictionary with download info
        """
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {}

    def _save_metadata(self, metadata: Dict[str, Any]):
        """
        Save metadata about the downloaded files.

        Parameters
        ----------
        metadata : dict
            Metadata to save
        """
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        except IOError as e:
            print(f"Warning: Could not save metadata: {e}")

    def _get_remote_info(self, url: str) -> Dict[str, Any]:
        """
        Get remote file information without downloading.

        Parameters
        ----------
        url : str
            URL to check

        Returns
        -------
        dict
            Dictionary with 'size', 'last_modified', and 'etag' if available
        """
        info = {}
        try:
            request = urllib.request.Request(url, method='HEAD')
            with urllib.request.urlopen(request) as response:
                headers = response.headers
                if 'Content-Length' in headers:
                    info['size'] = int(headers['Content-Length'])
                if 'Last-Modified' in headers:
                    info['last_modified'] = headers['Last-Modified']
                if 'ETag' in headers:
                    info['etag'] = headers['ETag'].strip('"')
        except (URLError, IOError):
            pass
        return info

    def _verify_checksum(self, filepath: Path, expected_checksum: str) -> bool:
        """
        Verify SHA256 checksum of a file.

        Parameters
        ----------
        filepath : Path
            Path to the file to verify
        expected_checksum : str
            Expected SHA256 checksum

        Returns
        -------
        bool
            True if checksum matches, False otherwise
        """
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest() == expected_checksum

    def _verify_multipole_structure(self, base_path: Path,
                                   show_progress: bool = True) -> bool:
        """
        Verify the expected multipole emulator structure.

        Expects folders like:
        - 0/, 2/, 4/  (monopole, quadrupole, hexadecapole)
        Each containing: 11/, loop/, ct/ subfolders

        Or the standard component structure:
        - 11/, loop/, ct/ (and optionally st/)

        Parameters
        ----------
        base_path : Path
            Path to check for emulator structure
        show_progress : bool
            Whether to show progress messages

        Returns
        -------
        bool
            True if valid structure found
        """
        # Check for numbered multipole folders (0, 2, 4)
        multipole_folders = ['0', '2', '4']

        # Check for standard component folders
        component_folders = ['11', 'loop', 'ct']

        # Look for multipole structure (0/, 2/, 4/)
        found_multipoles = []
        for mp in multipole_folders:
            if (base_path / mp).exists():
                # Check if it has the expected subfolders
                if all((base_path / mp / comp).exists() for comp in component_folders):
                    found_multipoles.append(mp)

        # Look for single component structure
        found_components = all((base_path / folder).exists() for folder in component_folders)

        if found_multipoles:
            if show_progress:
                print(f"✓ Found multipole folder structure: {', '.join(found_multipoles)}")
            return True
        elif found_components:
            if show_progress:
                print("✓ Found component folder structure (11/, loop/, ct/)")
            return True
        else:
            if show_progress:
                print("Warning: Expected folder structure not found")
                print("Looking for either:")
                print("  - Multipole folders: 0/, 2/, 4/ (each with 11/, loop/, ct/)")
                print("  - Component folders: 11/, loop/, ct/")

                # Debug: show what was actually found
                if base_path.exists():
                    items = list(base_path.iterdir())
                    if items:
                        print(f"Found in {base_path}:")
                        for item in items[:10]:
                            print(f"  - {item.name}")
            return False

    def download_and_extract(self, force: bool = False,
                           show_progress: bool = True) -> bool:
        """
        Download and extract emulator data if not already present.

        Parameters
        ----------
        force : bool
            Force re-download even if data exists
        show_progress : bool
            Whether to show progress

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        # Check if emulators are already extracted
        if not force and self.emulators_dir.exists():
            # Check for multipole structure
            multipole_folders = ['0', '2', '4']
            component_folders = ['11', 'loop', 'ct']

            all_multipoles_exist = True
            for mp in multipole_folders:
                mp_path = self.emulators_dir / mp
                if not mp_path.exists():
                    all_multipoles_exist = False
                    break
                # Check components
                for comp in component_folders:
                    if not (mp_path / comp).exists():
                        all_multipoles_exist = False
                        break
                if not all_multipoles_exist:
                    break

            if all_multipoles_exist:
                if show_progress:
                    print("Multipole emulator data already available.")
                return True

        # Download tar file if needed
        if force or not self.tar_path.exists():
            if show_progress:
                print(f"Downloading from Zenodo...")
            success = self._download_file(self.zenodo_url, self.tar_path,
                                         show_progress=show_progress)
            if not success:
                return False

            # Verify checksum if provided
            if self.expected_checksum:
                if show_progress:
                    print("Verifying checksum...")
                if not self._verify_checksum(self.tar_path, self.expected_checksum):
                    if show_progress:
                        print("ERROR: Checksum verification failed!")
                        print("The downloaded file may be corrupted.")
                    # Remove the corrupted file
                    if self.tar_path.exists():
                        self.tar_path.unlink()
                    return False
                elif show_progress:
                    print("✓ Checksum verified")

        # Extract tar file
        if show_progress:
            print("Extracting multipole emulator data...")

        # Create a temporary extraction directory
        temp_extract = self.cache_dir / "temp_extract"
        if temp_extract.exists():
            shutil.rmtree(temp_extract)

        success = self._extract_tar(self.tar_path, temp_extract,
                                   show_progress=show_progress)

        if success:
            # Find the directory containing multipole folders (0/, 2/, 4/)
            emulator_root = None

            # First, check if temp_extract itself has the multipole folders
            multipole_folders = ['0', '2', '4']
            if any((temp_extract / mp).exists() for mp in multipole_folders):
                emulator_root = temp_extract
            else:
                # Look for a subdirectory containing the multipole folders
                for item in temp_extract.iterdir():
                    if item.is_dir():
                        if any((item / mp).exists() for mp in multipole_folders):
                            emulator_root = item
                            break

            if emulator_root:
                # Create final destination if needed
                if self.emulators_dir.exists():
                    shutil.rmtree(self.emulators_dir)
                self.emulators_dir.mkdir(parents=True, exist_ok=True)

                # Copy each multipole folder to the final destination
                for mp in multipole_folders:
                    src_mp = emulator_root / mp
                    if src_mp.exists():
                        dest_mp = self.emulators_dir / mp
                        shutil.copytree(str(src_mp), str(dest_mp))
                        if show_progress:
                            print(f"  ✓ Copied multipole l={mp} emulator")

                # Clean up temp directory
                if temp_extract.exists():
                    shutil.rmtree(temp_extract)

                if show_progress:
                    print(f"✓ All multipole emulator data ready at: {self.emulators_dir}")

                # Save metadata about this download
                metadata = {
                    'downloaded_at': datetime.now().isoformat(),
                    'zenodo_url': self.zenodo_url,
                    'emulator_name': self.emulator_name,
                    'tar_file_size': self.tar_path.stat().st_size if self.tar_path.exists() else None,
                    'checksum_verified': self.expected_checksum is not None
                }
                # Get remote info if possible
                remote_info = self._get_remote_info(self.zenodo_url)
                if remote_info:
                    metadata['remote_info'] = remote_info
                self._save_metadata(metadata)

                return True
            else:
                if show_progress:
                    print("Error: Could not find multipole folders (0/, 2/, 4/) in extracted files")
                # Clean up
                if temp_extract.exists():
                    shutil.rmtree(temp_extract)
                return False

        return False

    def get_emulator_path(self, download_if_missing: bool = True) -> Optional[Path]:
        """
        Get the path to the emulator directory.

        Parameters
        ----------
        download_if_missing : bool
            Whether to download the data if not cached

        Returns
        -------
        Path or None
            Path to the emulator directory, or None if not available
        """
        if self.emulators_dir.exists():
            # Check if all multipole folders exist
            multipole_folders = ['0', '2', '4']
            if all((self.emulators_dir / mp).exists() for mp in multipole_folders):
                return self.emulators_dir

        # Download and extract if requested
        if download_if_missing:
            success = self.download_and_extract()
            if success and self.emulators_dir.exists():
                return self.emulators_dir

        return None

    def get_multipole_paths(self, download_if_missing: bool = True) -> Optional[Dict[int, Path]]:
        """
        Get paths to individual multipole emulator directories.

        Parameters
        ----------
        download_if_missing : bool
            Whether to download the data if not cached

        Returns
        -------
        dict or None
            Dictionary mapping multipole l values (0, 2, 4) to their paths,
            or None if not available
        """
        base_path = self.get_emulator_path(download_if_missing)
        if base_path is None:
            return None

        multipole_paths = {}
        for l in [0, 2, 4]:
            mp_path = base_path / str(l)
            if mp_path.exists():
                multipole_paths[l] = mp_path

        if len(multipole_paths) == 3:  # All three multipoles found
            return multipole_paths
        return None

    def clear_cache(self, clear_tar: bool = True, show_progress: bool = True):
        """
        Clear cached emulator files.

        Parameters
        ----------
        clear_tar : bool
            Whether to also clear the downloaded tar file
        show_progress : bool
            Whether to show progress messages
        """
        items_cleared = []

        # Clear extracted files
        if self.emulators_dir.exists():
            shutil.rmtree(self.emulators_dir)
            items_cleared.append("extracted emulators")

        # Clear tar file if requested
        if clear_tar and self.tar_path.exists():
            self.tar_path.unlink()
            items_cleared.append("tar archive")

        # Clear metadata
        if self.metadata_file.exists():
            self.metadata_file.unlink()
            items_cleared.append("metadata")

        if show_progress:
            if items_cleared:
                print(f"Cleared cached files: {', '.join(items_cleared)}")
            else:
                print("No cached files to clear")

    def check_for_updates(self, show_progress: bool = True) -> bool:
        """
        Check if updates are available for the cached emulators.

        Parameters
        ----------
        show_progress : bool
            Whether to show progress messages

        Returns
        -------
        bool
            True if updates are available, False otherwise
        """
        # Load cached metadata
        metadata = self._load_metadata()
        if not metadata:
            if show_progress:
                print("No cached metadata found")
            return True  # Assume update needed if no metadata

        # Get current remote info
        if show_progress:
            print("Checking for updates...")
        remote_info = self._get_remote_info(self.zenodo_url)

        if not remote_info:
            if show_progress:
                print("Could not check remote version")
            return False

        # Compare with cached info
        cached_remote = metadata.get('remote_info', {})
        update_available = False

        # Check ETag if available (most reliable)
        if 'etag' in remote_info and 'etag' in cached_remote:
            if remote_info['etag'] != cached_remote['etag']:
                update_available = True
                if show_progress:
                    print("✓ Update available: ETag changed")
        # Check file size
        elif 'size' in remote_info and 'size' in cached_remote:
            if remote_info['size'] != cached_remote['size']:
                update_available = True
                if show_progress:
                    print("✓ Update available: File size changed")
        # Check last modified
        elif 'last_modified' in remote_info and 'last_modified' in cached_remote:
            if remote_info['last_modified'] != cached_remote['last_modified']:
                update_available = True
                if show_progress:
                    print("✓ Update available: Last modified date changed")
        else:
            # Can't determine, assume no update
            if show_progress:
                print("Could not determine if updates are available")

        if not update_available and show_progress:
            print("✓ Cached version is up to date")

        return update_available

    def force_update(self, show_progress: bool = True) -> bool:
        """
        Force update by clearing cache and re-downloading.

        Parameters
        ----------
        show_progress : bool
            Whether to show progress messages

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if show_progress:
            print("Force updating multipole emulator data...")

        # Clear entire cache directory and recreate it
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            if show_progress:
                print(f"Cleared cache directory: {self.cache_dir}")

        # Recreate the cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if show_progress:
            print(f"Recreated cache directory: {self.cache_dir}")

        # Re-download and extract
        if show_progress:
            print("Re-downloading latest version...")
        return self.download_and_extract(force=True, show_progress=show_progress)

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached files.

        Returns
        -------
        dict
            Dictionary with cache information
        """
        info = {
            'cache_dir': str(self.cache_dir),
            'emulator_name': self.emulator_name,
            'has_cached_data': self.emulators_dir.exists()
        }

        # Add size information
        if self.emulators_dir.exists():
            total_size = sum(f.stat().st_size for f in self.emulators_dir.rglob('*') if f.is_file())
            info['extracted_size_mb'] = round(total_size / (1024 * 1024), 2)

        if self.tar_path.exists():
            info['tar_size_mb'] = round(self.tar_path.stat().st_size / (1024 * 1024), 2)

        # Add metadata info
        metadata = self._load_metadata()
        if metadata:
            info['downloaded_at'] = metadata.get('downloaded_at')
            info['checksum_verified'] = metadata.get('checksum_verified', False)

        return info


# Convenience functions for direct access
_default_fetcher = None


def get_fetcher(zenodo_url: str = None,
                emulator_name: str = None,
                cache_dir: Optional[Union[str, Path]] = None,
                expected_checksum: str = None) -> MultipoleDataFetcher:
    """
    Get the default fetcher instance (singleton pattern).

    Parameters
    ----------
    zenodo_url : str, optional
        URL to download the emulator tar.gz file from.
        If None, uses the default pybird mnuw0wacdm URL.
    emulator_name : str, optional
        Name identifier for the emulator set.
        If None, uses "pybird_mnuw0wacdm".
    cache_dir : str or Path, optional
        Cache directory for the fetcher
    expected_checksum : str, optional
        Expected SHA256 checksum of the downloaded file.

    Returns
    -------
    MultipoleDataFetcher
        The fetcher instance
    """
    global _default_fetcher

    # Use defaults if not specified
    if zenodo_url is None:
        zenodo_url = "https://zenodo.org/records/17172502/files/trained_effort_pybird_mnuw0wacdm.tar.gz?download=1"
    if emulator_name is None:
        emulator_name = "pybird_mnuw0wacdm"

    if _default_fetcher is None:
        _default_fetcher = MultipoleDataFetcher(zenodo_url, emulator_name, cache_dir, expected_checksum)
    return _default_fetcher


def get_emulator_path() -> Optional[Path]:
    """
    Get the path to the multipole emulator directory.

    Returns
    -------
    Path or None
        Path to the emulator directory
    """
    return get_fetcher().get_emulator_path()


def get_multipole_paths() -> Optional[Dict[int, Path]]:
    """
    Get paths to individual multipole emulator directories.

    Returns
    -------
    dict or None
        Dictionary mapping multipole l values (0, 2, 4) to their paths
    """
    return get_fetcher().get_multipole_paths()


def clear_cache(model_name: str = None, clear_tar: bool = True, show_progress: bool = True):
    """
    Clear cached emulator files for a specific model or all models.

    Parameters
    ----------
    model_name : str, optional
        Specific model to clear cache for. If None, clears default model.
    clear_tar : bool
        Whether to also clear downloaded tar files
    show_progress : bool
        Whether to show progress messages

    Examples
    --------
    >>> import jaxeffort
    >>> # Clear cache for default model
    >>> jaxeffort.clear_cache()
    >>> # Clear cache but keep tar file
    >>> jaxeffort.clear_cache(clear_tar=False)
    """
    if model_name:
        # Clear specific model
        zenodo_url = f"https://zenodo.org/records/17138475/files/trained_effort_{model_name}.tar.gz?download=1"
        fetcher = MultipoleDataFetcher(zenodo_url, model_name)
    else:
        fetcher = get_fetcher()

    fetcher.clear_cache(clear_tar=clear_tar, show_progress=show_progress)


def check_for_updates(model_name: str = None, show_progress: bool = True) -> bool:
    """
    Check if updates are available for cached emulators.

    Parameters
    ----------
    model_name : str, optional
        Specific model to check. If None, checks default model.
    show_progress : bool
        Whether to show progress messages

    Returns
    -------
    bool
        True if updates are available, False otherwise

    Examples
    --------
    >>> import jaxeffort
    >>> if jaxeffort.check_for_updates():
    ...     print("Updates available!")
    ...     jaxeffort.force_update()
    """
    if model_name:
        zenodo_url = f"https://zenodo.org/records/17138475/files/trained_effort_{model_name}.tar.gz?download=1"
        fetcher = MultipoleDataFetcher(zenodo_url, model_name)
    else:
        fetcher = get_fetcher()

    return fetcher.check_for_updates(show_progress=show_progress)


def force_update(model_name: str = None, show_progress: bool = True) -> bool:
    """
    Force update emulators by clearing cache and re-downloading.

    This function will:
    1. Clear the entire cache directory (~/.jaxeffort_data)
    2. Recreate the cache directory
    3. Download the latest version from Zenodo
    4. Extract and set up the emulators

    Parameters
    ----------
    model_name : str, optional
        Specific model to update. If None, updates default model.
        Note: This will still clear the entire cache directory.
    show_progress : bool
        Whether to show progress messages

    Returns
    -------
    bool
        True if successful, False otherwise

    Examples
    --------
    >>> import jaxeffort
    >>> # Force update to get latest version (clears entire cache)
    >>> jaxeffort.force_update()
    >>> # Update specific model (still clears entire cache)
    >>> jaxeffort.force_update("pybird_mnuw0wacdm")
    """
    if model_name:
        zenodo_url = f"https://zenodo.org/records/17138475/files/trained_effort_{model_name}.tar.gz?download=1"
        fetcher = MultipoleDataFetcher(zenodo_url, model_name)
    else:
        fetcher = get_fetcher()

    return fetcher.force_update(show_progress=show_progress)


def get_cache_info(model_name: str = None) -> Dict[str, Any]:
    """
    Get information about cached emulator files.

    Parameters
    ----------
    model_name : str, optional
        Specific model to get info for. If None, gets info for default model.

    Returns
    -------
    dict
        Dictionary with cache information including:
        - cache_dir: Path to cache directory
        - emulator_name: Name of the emulator
        - has_cached_data: Whether data is cached
        - extracted_size_mb: Size of extracted files in MB
        - tar_size_mb: Size of tar file in MB
        - downloaded_at: When the data was downloaded
        - checksum_verified: Whether checksum was verified

    Examples
    --------
    >>> import jaxeffort
    >>> info = jaxeffort.get_cache_info()
    >>> print(f"Cache location: {info['cache_dir']}")
    >>> print(f"Downloaded at: {info['downloaded_at']}")
    """
    if model_name:
        zenodo_url = f"https://zenodo.org/records/17138475/files/trained_effort_{model_name}.tar.gz?download=1"
        fetcher = MultipoleDataFetcher(zenodo_url, model_name)
    else:
        fetcher = get_fetcher()

    return fetcher.get_cache_info()


def clear_all_cache(show_progress: bool = True):
    """
    Clear ALL cached jaxeffort data.

    This will remove the entire ~/.jaxeffort_data directory.

    Parameters
    ----------
    show_progress : bool
        Whether to show progress messages

    Examples
    --------
    >>> import jaxeffort
    >>> # Remove all cached data
    >>> jaxeffort.clear_all_cache()
    """
    cache_dir = Path.home() / ".jaxeffort_data"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        if show_progress:
            print(f"Cleared all cached data from {cache_dir}")
    else:
        if show_progress:
            print("No cached data to clear")
