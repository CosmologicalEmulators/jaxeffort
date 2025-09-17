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
    MultipoleDataFetcher
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
        "zenodo_url": "https://zenodo.org/records/17138352/files/trained_effort_pybird_mnuw0wacdm.tar.gz?download=1",
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
        Dictionary containing the loaded multipole emulator
    """
    emulator_dict = {}

    try:
        # Initialize fetcher for this model
        fetcher = get_fetcher(
            zenodo_url=config["zenodo_url"],
            emulator_name=model_name,
            expected_checksum=config.get("checksum")
        )

        # Download if needed and requested
        if auto_download:
            emulator_path = fetcher.get_emulator_path(download_if_missing=True)
        else:
            emulator_path = fetcher.get_emulator_path(download_if_missing=False)

        if emulator_path and emulator_path.exists():
            # Load the appropriate emulator type
            try:
                if config.get("has_noise", False):
                    # Load multipole emulator with noise component
                    emulator_dict["multipole_noise"] = load_multipole_noise_emulator(str(emulator_path))
                    emulator_dict["type"] = "multipole_noise"
                else:
                    # Load standard multipole emulator
                    emulator_dict["multipole"] = load_multipole_emulator(str(emulator_path))
                    emulator_dict["type"] = "multipole"

                emulator_dict["path"] = str(emulator_path)
                emulator_dict["loaded"] = True

            except Exception as e:
                warnings.warn(f"Error loading emulators from {emulator_path}: {e}")
                emulator_dict["loaded"] = False
                emulator_dict["error"] = str(e)
        else:
            warnings.warn(f"Could not find emulator data for {model_name}")
            emulator_dict["loaded"] = False

    except Exception as e:
        warnings.warn(f"Could not initialize {model_name}: {e}")
        emulator_dict["loaded"] = False
        emulator_dict["error"] = str(e)

    return emulator_dict


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
            if trained_emulators[model_name].get("loaded"):
                emulator_type = trained_emulators[model_name].get("type", "unknown")
                print(f"  ✓ {model_name}: Loaded {emulator_type} emulator")
            else:
                warnings.warn(f"Failed to load emulator for {model_name}")

        except Exception as e:
            # Ensure import doesn't fail completely
            warnings.warn(f"Failed to load {model_name} emulators: {e}")
            trained_emulators[model_name] = {"loaded": False, "error": str(e)}
else:
    # Create empty structure when auto-download is disabled
    for model_name, config in EMULATOR_CONFIGS.items():
        trained_emulators[model_name] = {"loaded": False, "disabled": True}


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
        if trained_emulators[model_name].get("loaded"):
            emulator_type = trained_emulators[model_name].get("type", "unknown")
            print(f"  ✓ Loaded {emulator_type} emulator")
        else:
            print(f"  ✗ Failed to load emulator")
    else:
        # Create empty structure
        trained_emulators[model_name] = {"loaded": False}

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


def get_default_emulator():
    """
    Get the default loaded multipole emulator.

    Returns
    -------
    MultipoleEmulators or MultipoleNoiseEmulator
        The default loaded emulator, or None if not loaded
    """
    # Return the first successfully loaded emulator
    for model_name, emulator_data in trained_emulators.items():
        if emulator_data.get("loaded"):
            if emulator_data["type"] == "multipole_noise":
                return emulator_data["multipole_noise"]
            elif emulator_data["type"] == "multipole":
                return emulator_data["multipole"]
    return None