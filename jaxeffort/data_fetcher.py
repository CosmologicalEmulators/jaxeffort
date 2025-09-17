"""
Data fetcher for jaxeffort emulator files from Zenodo.

This module handles downloading, extracting, and caching of trained multipole emulator data.
Based on the jaxcapse data fetcher design but adapted for jaxeffort's multipole structure.
"""

import hashlib
import os
import shutil
import tarfile
import urllib.request
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
        - monoquad_l0l2/  (monopole and quadrupole)
        - hexa_l4/ (hexadecapole, if present)

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
        # Check for multipole-named folders
        multipole_folders = ['monoquad_l0l2', 'hexa_l4', 'l0l2', 'l4']

        # Check for standard component folders
        component_folders = ['11', 'loop', 'ct']

        # Look for either multipole or component structure
        found_multipole = any((base_path / folder).exists() for folder in multipole_folders)
        found_components = all((base_path / folder).exists() for folder in component_folders)

        if found_multipole:
            if show_progress:
                print("✓ Found multipole folder structure")
            return True
        elif found_components:
            if show_progress:
                print("✓ Found component folder structure (11/, loop/, ct/)")
            return True
        else:
            if show_progress:
                print("Warning: Expected folder structure not found")
                print("Looking for either:")
                print("  - Multipole folders: monoquad_l0l2/, hexa_l4/")
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
            if self._verify_multipole_structure(self.emulators_dir, show_progress=False):
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
            # Find the actual emulator directory in the extracted files
            # It might be nested or have a different name
            emulator_root = None

            # Look for the expected structure in extracted files
            for item in temp_extract.rglob("*"):
                if item.is_dir():
                    if self._verify_multipole_structure(item, show_progress=False):
                        emulator_root = item
                        break

            if emulator_root:
                # Move to final destination
                if self.emulators_dir.exists():
                    shutil.rmtree(self.emulators_dir)
                shutil.move(str(emulator_root), str(self.emulators_dir))

                # Clean up temp directory
                if temp_extract.exists():
                    shutil.rmtree(temp_extract)

                if show_progress:
                    print(f"✓ Multipole emulator data ready at: {self.emulators_dir}")
                return True
            else:
                if show_progress:
                    print("Error: Could not find valid emulator structure in extracted files")
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
        if self.emulators_dir.exists() and self._verify_multipole_structure(self.emulators_dir, show_progress=False):
            return self.emulators_dir

        # Download and extract if requested
        if download_if_missing:
            success = self.download_and_extract()
            if success and self.emulators_dir.exists():
                return self.emulators_dir

        return None

    def clear_cache(self):
        """
        Clear cached emulator files.
        """
        # Clear extracted files
        if self.emulators_dir.exists():
            shutil.rmtree(self.emulators_dir)
        # Clear tar file
        if self.tar_path.exists():
            self.tar_path.unlink()
        print("Cleared cached multipole emulator files")


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

    # Use defaults for get_fetcher to maintain backward compatibility
    if zenodo_url is None:
        zenodo_url = "https://zenodo.org/records/17138352/files/trained_effort_pybird_mnuw0wacdm.tar.gz?download=1"
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