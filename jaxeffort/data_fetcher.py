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
        zenodo_url = "https://zenodo.org/records/17138475/files/trained_effort_pybird_mnuw0wacdm.tar.gz?download=1"
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