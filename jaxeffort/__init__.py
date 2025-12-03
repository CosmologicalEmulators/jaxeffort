"""
jaxeffort: JAX-based Effective Field Theory for Galaxy Power Spectrum

This package provides tools for emulating galaxy power spectra using JAX,
with automatic downloading and caching of pretrained multipole emulators.

Emulator configurations are defined in Artifacts.toml (similar to Julia's Pkg.Artifacts).
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import core functionality explicitly (not using *)
from jaxeffort.jaxeffort import (
    MLP,
    MultipoleEmulators,
    load_multipole_emulator,
    load_component_emulator,
    load_bias_combination,
    load_jacobian_bias_combination,
    load_preprocessing,
    load_stoch_model,
    # Cosmology functions from jaxace (re-exported for convenience)
    w0waCDMCosmology,
    a_z,
    E_a,
    E_z,
    dlogEdloga,
    Ωm_a,
    D_z,
    f_z,
    D_f_z,
    r_z,
    dA_z,
    dL_z,
    ρc_z,
    Ωtot_z,
    F,
    dFdy,
    ΩνE2,
    growth_solver,
    growth_ode_system,
    # Neural network infrastructure from jaxace
    init_emulator,
    FlaxEmulator,
    maximin,
    inv_maximin,
)

# Import fetch_artifacts for artifact management
from fetch_artifacts import (
    load_artifacts,
    artifact,
    artifact_exists,
    clear_artifact_cache,
    get_cache_dir,
    set_cache_dir,
    bind_artifact,
    add_artifact,
)

# Initialize the trained_emulators dictionary
trained_emulators: Dict[str, Dict[str, Optional[MultipoleEmulators]]] = {}

# Path to Artifacts.toml (in package directory)
_ARTIFACTS_TOML = Path(__file__).parent.parent / "Artifacts.toml"

# Global artifact manager
_artifact_manager = None


def _get_artifact_manager():
    """Get or create the artifact manager singleton."""
    global _artifact_manager
    if _artifact_manager is None:
        if _ARTIFACTS_TOML.exists():
            _artifact_manager = load_artifacts(_ARTIFACTS_TOML)
        else:
            warnings.warn(f"Artifacts.toml not found at {_ARTIFACTS_TOML}")
    return _artifact_manager


def list_emulators() -> List[str]:
    """
    List all available emulators defined in Artifacts.toml.

    Returns
    -------
    list of str
        Names of available emulators.
    """
    manager = _get_artifact_manager()
    if manager is None:
        return []
    return list(manager.artifacts.keys())


def get_emulator_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about an emulator from Artifacts.toml.

    Parameters
    ----------
    model_name : str
        Name of the emulator.

    Returns
    -------
    dict
        Emulator information including description, has_noise, etc.
    """
    manager = _get_artifact_manager()
    if manager is None:
        raise RuntimeError("Artifacts.toml not found")

    if model_name not in manager:
        raise ValueError(f"Model '{model_name}' not found. Available: {list_emulators()}")

    entry = manager.artifacts[model_name]
    return {
        "name": model_name,
        "git_tree_sha1": entry.git_tree_sha1,
        "lazy": entry.lazy,
        "cached": manager.exists(model_name),
        **entry.metadata,  # Includes description, has_noise, etc.
    }


__all__ = [
    # Core emulator classes
    "MLP",
    "MultipoleEmulators",
    # Loading functions
    "load_multipole_emulator",
    "load_component_emulator",
    "load_bias_combination",
    "load_jacobian_bias_combination",
    "load_preprocessing",
    "load_stoch_model",
    # Artifact management
    "get_emulator_path",
    "list_emulators",
    "get_emulator_info",
    "clear_cache",
    "get_cache_info",
    "clear_all_cache",
    "add_emulator",
    # Trained emulators dictionary
    "trained_emulators",
    "reload_emulators",
    # Cosmology functions (from jaxace)
    "w0waCDMCosmology",
    "a_z",
    "E_a",
    "E_z",
    "dlogEdloga",
    "Ωm_a",
    "D_z",
    "f_z",
    "D_f_z",
    "r_z",
    "dA_z",
    "dL_z",
    "ρc_z",
    "Ωtot_z",
    "F",
    "dFdy",
    "ΩνE2",
    "growth_solver",
    "growth_ode_system",
    # Neural network infrastructure (from jaxace)
    "init_emulator",
    "FlaxEmulator",
    "maximin",
    "inv_maximin",
]

__version__ = "0.3.0"


def get_emulator_path(model_name: str = None, download_if_missing: bool = True) -> Path:
    """
    Get the path to an emulator's data directory.

    Parameters
    ----------
    model_name : str, optional
        Name of the model. If None, returns path for default model.
    download_if_missing : bool
        Whether to download if not cached. Default: True.

    Returns
    -------
    Path
        Path to the emulator directory.
    """
    if model_name is None:
        model_name = "pybird_mnuw0wacdm"

    manager = _get_artifact_manager()
    if manager is None:
        raise RuntimeError("Artifact manager not initialized. Artifacts.toml not found.")

    if model_name not in manager:
        raise ValueError(
            f"Model '{model_name}' not found. Available: {list_emulators()}"
        )

    return manager.get_path(model_name)


def clear_cache(model_name: str = None):
    """
    Clear cached emulator files.

    Parameters
    ----------
    model_name : str, optional
        Specific model to clear. If None, clears all.
    """
    manager = _get_artifact_manager()
    if manager is None:
        return

    if model_name:
        manager.clear(model_name)
    else:
        manager.clear()


def clear_all_cache():
    """Clear ALL cached jaxeffort data."""
    clear_cache()


def get_cache_info(model_name: str = None) -> dict:
    """
    Get information about cached emulator files.

    Parameters
    ----------
    model_name : str, optional
        Specific model to get info for.

    Returns
    -------
    dict
        Cache information.
    """
    manager = _get_artifact_manager()
    if manager is None:
        return {"error": "Artifact manager not initialized"}

    if model_name is None:
        model_name = "pybird_mnuw0wacdm"

    return {
        "cache_dir": str(get_cache_dir()),
        "emulator_name": model_name,
        "has_cached_data": manager.exists(model_name),
    }


def _load_emulator_set(model_name: str, auto_download: bool = True):
    """
    Load a set of multipole emulators for a given model.

    Parameters
    ----------
    model_name : str
        Name of the model (e.g., "pybird_mnuw0wacdm")
    auto_download : bool
        Whether to automatically download if not cached

    Returns
    -------
    dict
        Dictionary with string keys ("0", "2", "4") mapping to multipole emulators
    """
    emulators = {}

    try:
        manager = _get_artifact_manager()
        if manager is None or model_name not in manager:
            warnings.warn(f"Model '{model_name}' not found in Artifacts.toml")
            return {"0": None, "2": None, "4": None}

        # Check if we should download
        if not auto_download and not manager.exists(model_name):
            return {"0": None, "2": None, "4": None}

        # Get path (downloads if needed)
        emulator_path = manager.get_path(model_name)

        if emulator_path and emulator_path.exists():
            # Load each multipole emulator
            for l in [0, 2, 4]:
                mp_path = emulator_path / str(l)
                if mp_path.exists():
                    try:
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
            warnings.warn(f"Could not find emulator data for {model_name}")
            emulators = {"0": None, "2": None, "4": None}

    except Exception as e:
        warnings.warn(f"Could not initialize {model_name}: {e}")
        emulators = {"0": None, "2": None, "4": None}

    return emulators


def add_emulator(
    model_name: str,
    tarball_url: str,
    description: str = None,
    has_noise: bool = False,
    force: bool = False,
    auto_load: bool = True,
):
    """
    Add a new emulator by downloading from URL and adding to Artifacts.toml.

    This is similar to Julia's ArtifactUtils.add_artifact!().

    Parameters
    ----------
    model_name : str
        Name for the model
    tarball_url : str
        URL to download the emulator tarball
    description : str, optional
        Description of the model
    has_noise : bool, optional
        Whether the emulator includes noise component
    force : bool, optional
        Overwrite if already exists
    auto_load : bool, optional
        Whether to immediately load the emulators

    Returns
    -------
    dict
        The loaded emulator for this model
    """
    global trained_emulators, _artifact_manager

    # Use pyartifacts.add_artifact to download and add to TOML
    result = add_artifact(
        toml_path=_ARTIFACTS_TOML,
        name=model_name,
        tarball_url=tarball_url,
        lazy=True,
        force=force,
        verbose=True,
    )

    # Add extra metadata to the TOML
    # We need to update the TOML with description and has_noise
    try:
        import tomlkit
        with open(_ARTIFACTS_TOML, "r") as f:
            doc = tomlkit.load(f)

        if model_name in doc:
            if description:
                doc[model_name]["description"] = description
            doc[model_name]["has_noise"] = has_noise

            with open(_ARTIFACTS_TOML, "w") as f:
                tomlkit.dump(doc, f)
    except ImportError:
        warnings.warn("tomlkit not installed, cannot add metadata to Artifacts.toml")

    # Reset artifact manager to reload the TOML
    _artifact_manager = None

    # Load if requested
    if auto_load:
        trained_emulators[model_name] = _load_emulator_set(model_name, auto_download=True)

        loaded = sum(1 for v in trained_emulators[model_name].values() if v is not None)
        if loaded == 0:
            warnings.warn(f"Failed to load emulator for {model_name}")
    else:
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

    manager = _get_artifact_manager()
    if manager is None:
        warnings.warn("Cannot reload: Artifacts.toml not found")
        return trained_emulators

    if model_name:
        if model_name in manager:
            trained_emulators[model_name] = _load_emulator_set(model_name, auto_download=True)
        else:
            raise ValueError(
                f"Unknown model: {model_name}. Available: {list_emulators()}"
            )
    else:
        # Reload all models from Artifacts.toml
        for name in manager.artifacts:
            trained_emulators[name] = _load_emulator_set(name, auto_download=True)

    return trained_emulators


# Load emulators on import (unless disabled)
if not os.environ.get("JAXEFFORT_NO_AUTO_DOWNLOAD"):
    manager = _get_artifact_manager()
    if manager is not None:
        for model_name in manager.artifacts:
            try:
                trained_emulators[model_name] = _load_emulator_set(
                    model_name, auto_download=True
                )

                loaded = sum(1 for v in trained_emulators[model_name].values() if v is not None)
                if loaded == 0:
                    warnings.warn(f"Failed to load any multipoles for {model_name}")

            except Exception as e:
                warnings.warn(f"Failed to load {model_name} emulators: {e}")
                trained_emulators[model_name] = {"0": None, "2": None, "4": None}
else:
    # Create empty structure when auto-download is disabled
    manager = _get_artifact_manager()
    if manager is not None:
        for model_name in manager.artifacts:
            trained_emulators[model_name] = {"0": None, "2": None, "4": None}
