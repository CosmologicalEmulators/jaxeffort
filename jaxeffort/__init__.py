"""
jaxeffort: JAX-based Effective Field Theory for Galaxy Power Spectrum

This package provides tools for emulating galaxy power spectra using JAX,
with automatic downloading and caching of pretrained multipole emulators.
"""

import os
import warnings
from pathlib import Path

# Import core functionality
from jaxeffort.jaxeffort import *

# Import data fetcher functionality
from .data_fetcher import (
    get_emulator_path,
    get_fetcher,
    MultipoleDataFetcher,
    clear_cache,
    check_for_updates,
    force_update,
    get_cache_info,
    clear_all_cache
)

# Explicitly export stochastic term function
from jaxeffort.jaxeffort import get_stoch_terms

# Import the loading functions we'll use
from jaxeffort.jaxeffort import (
    load_multipole_emulator,
    load_multipole_noise_emulator,
)

__all__ = [
    # Core emulator classes
    "MLP",
    "MultipoleEmulators",
    "MultipoleNoiseEmulator",
    # Loading functions
    "load_multipole_emulator",
    "load_multipole_noise_emulator",
    "get_stoch_terms",
    # Data fetcher
    "get_emulator_path",
    "get_fetcher",
    "MultipoleDataFetcher",
    # Cache management
    "clear_cache",
    "check_for_updates",
    "force_update",
    "get_cache_info",
    "clear_all_cache",
    # Trained emulators dictionary
    "trained_emulators",
    "EMULATOR_CONFIGS",
    "add_emulator_config",
    "reload_emulators",
]

__version__ = "0.1.0"

# Initialize the trained_emulators dictionary
trained_emulators = {}

# Define available emulator configurations
# This can be easily extended with new models in the future
EMULATOR_CONFIGS = {
    "pybird_mnuw0wacdm": {
        "zenodo_url": "https://zenodo.org/records/17172502/files/trained_effort_pybird_mnuw0wacdm.tar.gz?download=1",
        "description": "PyBird emulator for massive neutrinos, w0wa CDM cosmology",
        "has_noise": False,  # Set to True if the emulator includes noise (st/) component
    }
    # Future models can be added here:
    # "camb_lcdm": {
    #     "zenodo_url": "https://zenodo.org/...",
    #     "description": "CAMB-based LCDM model",
    #     "has_noise": True,
    # }
}


def _load_emulator_set(model_name: str, config: dict, auto_download: bool = True):
    """
    Helper function to load a set of multipole emulators for a given model.

    Parameters
    ----------
    model_name : str
        Name of the model (e.g., "pybird_mnuw0wacdm")
    config : dict
        Configuration dictionary with zenodo_url and other settings
    auto_download : bool
        Whether to automatically download if not cached

    Returns
    -------
    dict
        Dictionary with string keys ("0", "2", "4") mapping to multipole emulators
    """
    emulators = {}

    try:
        # Initialize fetcher for this model
        fetcher = get_fetcher(
            zenodo_url=config["zenodo_url"],
            emulator_name=model_name,
            expected_checksum=config.get("checksum")
        )

        # Get multipole paths
        multipole_paths = fetcher.get_multipole_paths(download_if_missing=auto_download)

        if multipole_paths:
            # Load each multipole emulator with string keys
            for l, mp_path in multipole_paths.items():
                if mp_path and mp_path.exists():
                    try:
                        if config.get("has_noise", False):
                            # Load multipole emulator with noise component
                            emulators[str(l)] = load_multipole_noise_emulator(str(mp_path))
                        else:
                            # Load standard multipole emulator
                            emulators[str(l)] = load_multipole_emulator(str(mp_path))
                    except Exception as e:
                        emulators[str(l)] = None
                        warnings.warn(f"Error loading multipole l={l} from {mp_path}: {e}")
                else:
                    emulators[str(l)] = None

            loaded = sum(1 for v in emulators.values() if v is not None)
            if loaded == 0:
                warnings.warn(f"No multipole emulators loaded for {model_name}")
        else:
            warnings.warn(f"Could not find multipole emulator data for {model_name}")
            # Create empty entries for expected multipoles
            emulators = {"0": None, "2": None, "4": None}

    except Exception as e:
        warnings.warn(f"Could not initialize {model_name}: {e}")
        # Create empty entries for expected multipoles
        emulators = {"0": None, "2": None, "4": None}

    return emulators


# Load default emulators on import (unless disabled)
if not os.environ.get("JAXEFFORT_NO_AUTO_DOWNLOAD"):
    print("jaxeffort: Initializing multipole emulators...")

    # Load all configured models
    for model_name, config in EMULATOR_CONFIGS.items():
        try:
            print(f"  Loading {model_name}...")
            trained_emulators[model_name] = _load_emulator_set(
                model_name,
                config,
                auto_download=True
            )

            # Report loading status
            loaded = sum(1 for v in trained_emulators[model_name].values() if v is not None)
            total = 3  # Expecting multipoles 0, 2, 4
            if loaded > 0:
                loaded_ls = [k for k, v in trained_emulators[model_name].items() if v is not None]
                print(f"  {model_name}: Loaded {loaded}/{total} multipoles (l={loaded_ls})")
            else:
                warnings.warn(f"Failed to load any multipoles for {model_name}")

        except Exception as e:
            # Ensure import doesn't fail completely
            warnings.warn(f"Failed to load {model_name} emulators: {e}")
            trained_emulators[model_name] = {"0": None, "2": None, "4": None}
else:
    # Create empty structure when auto-download is disabled
    for model_name, config in EMULATOR_CONFIGS.items():
        trained_emulators[model_name] = {"0": None, "2": None, "4": None}


def add_emulator_config(model_name: str,
                        zenodo_url: str,
                        description: str = None,
                        has_noise: bool = False,
                        checksum: str = None,
                        auto_load: bool = True):
    """
    Add a new emulator configuration and optionally load it.

    Parameters
    ----------
    model_name : str
        Name for the model (e.g., "camb_lcdm")
    zenodo_url : str
        URL to download the emulator tar.gz file from
    description : str, optional
        Description of the model
    has_noise : bool, optional
        Whether the emulator includes noise component (st/ folder)
    checksum : str, optional
        Expected SHA256 checksum of the downloaded file
    auto_load : bool, optional
        Whether to immediately load the emulators

    Returns
    -------
    dict
        The loaded emulator for this model
    """
    global EMULATOR_CONFIGS, trained_emulators

    # Add to configuration
    EMULATOR_CONFIGS[model_name] = {
        "zenodo_url": zenodo_url,
        "description": description or f"{model_name} emulators",
        "has_noise": has_noise
    }

    # Add checksum if provided
    if checksum:
        EMULATOR_CONFIGS[model_name]["checksum"] = checksum

    # Load if requested
    if auto_load:
        print(f"Loading {model_name} emulator...")
        trained_emulators[model_name] = _load_emulator_set(
            model_name,
            EMULATOR_CONFIGS[model_name],
            auto_download=True
        )

        # Report status
        loaded = sum(1 for v in trained_emulators[model_name].values() if v is not None)
        if loaded > 0:
            print(f"  ✓ Loaded {loaded}/3 multipoles")
        else:
            print(f"  ✗ Failed to load emulator")
    else:
        # Create empty structure
        trained_emulators[model_name] = {"0": None, "2": None, "4": None}

    return trained_emulators[model_name]


def reload_emulators(model_name: str = None):
    """
    Reload emulators for a specific model or all models.

    Parameters
    ----------
    model_name : str, optional
        Specific model to reload. If None, reloads all.

    Returns
    -------
    dict
        The trained_emulators dictionary
    """
    global trained_emulators

    if model_name:
        # Reload specific model
        if model_name in EMULATOR_CONFIGS:
            print(f"Reloading {model_name}...")
            trained_emulators[model_name] = _load_emulator_set(
                model_name,
                EMULATOR_CONFIGS[model_name],
                auto_download=True
            )
        else:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(EMULATOR_CONFIGS.keys())}")
    else:
        # Reload all models
        print("Reloading all emulators...")
        for name, config in EMULATOR_CONFIGS.items():
            trained_emulators[name] = _load_emulator_set(name, config, auto_download=True)

    return trained_emulators
