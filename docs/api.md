# API Reference

## Core Classes

### MultipoleEmulators

The main class for computing galaxy power spectrum multipoles.

**Methods:**
- `get_Pl(cosmo_params, bias_params, D)`: Compute power spectrum multipole
- `P11`: Access P11 component emulator
- `Ploop`: Access Ploop component emulator
- `Pct`: Access Pct component emulator

### MultipoleNoiseEmulator

Extension of MultipoleEmulators with stochastic noise terms.

**Methods:**
- `get_Pl(cosmo_params, bias_params, D, stoch_params)`: Compute multipole with noise

## Cosmology Classes

### w0waCDMCosmology

w₀wₐCDM cosmology with massive neutrinos support.

**Parameters:**
- `ln10As`: Log amplitude of primordial power spectrum
- `ns`: Spectral index
- `h`: Hubble parameter
- `omega_b`: Baryon density
- `omega_c`: CDM density
- `m_nu`: Neutrino mass [eV]
- `w0`: Dark energy equation of state
- `wa`: Dark energy equation of state evolution

**Methods:**
- `D_z(z)`: Growth factor at redshift z
- `H_z(z)`: Hubble parameter at redshift z
- `comoving_distance(z)`: Comoving distance to redshift z

## Loading Functions

### load_multipole_emulator

Load a multipole emulator from disk.

```python
emulator = jaxeffort.load_multipole_emulator(path)
```

**Parameters:**
- `path`: Path to emulator directory

**Returns:**
- `MultipoleEmulators` instance

### force_update

Force update of cached emulator data.

```python
jaxeffort.force_update()
```

Downloads latest emulator data from Zenodo and updates local cache.

## Pre-trained Emulators

The following pre-trained emulators are available through `jaxeffort.trained_emulators`:

### pybird_mnuw0wacdm

Emulator trained on PyBird calculations with w₀wₐCDM cosmology and massive neutrinos.

- **Multipoles available**: 0 (monopole), 2 (quadrupole), 4 (hexadecapole)
- **Redshift range**: 0.5 - 2.0
- **k range**: 0.005 - 0.3 h/Mpc
- **Parameters**:
  - Cosmological: z, ln10As, ns, H0, ombh2, omch2, Mnu, w0, wa
  - Bias: b1, b2, b3, b4, b5, b6, b7, f

Example usage:
```python
P0 = jaxeffort.trained_emulators["pybird_mnuw0wacdm"]["0"]
P2 = jaxeffort.trained_emulators["pybird_mnuw0wacdm"]["2"]
P4 = jaxeffort.trained_emulators["pybird_mnuw0wacdm"]["4"]
```

## Utility Functions

### clear_cache

Clear the local cache of downloaded emulator data.

```python
jaxeffort.clear_cache()
```

Removes all cached emulator files from `~/.jaxeffort_data/`.