# Data Management

This guide explains how to manage emulator data in jaxeffort.

## Default Data Location

jaxeffort stores emulator data in a cache directory:

- **Linux/Mac**: `~/.jaxeffort_data/`
- **Windows**: `%USERPROFILE%\.jaxeffort_data\`

## Automatic Data Download

When you first use jaxeffort, it automatically downloads the required emulator data:

```python
import jaxeffort

# This triggers automatic download on first use
P0 = jaxeffort.trained_emulators["pybird_mnuw0wacdm"]["0"]
```

## Force Update

To ensure you have the latest version of the emulator data:

```python
import jaxeffort

# Force re-download of emulator data
jaxeffort.force_update()
```

This will:
1. Clear the existing cache directory
2. Download the latest emulator data from Zenodo
3. Extract and organize the data

## Loading Custom Emulators

You can load emulators from custom locations:

```python
import jaxeffort

# Load from a specific directory
custom_emulator = jaxeffort.load_multipole_emulator("/path/to/your/emulator/")
```

### Expected Directory Structure

Custom emulator directories should have the following structure:

```
emulator_directory/
├── 11/                  # P11 component
│   ├── weights.npy      # Neural network weights
│   ├── inminmax.npy     # Input normalization
│   ├── outminmax.npy    # Output normalization
│   ├── nn_setup.json    # Network architecture
│   ├── k.npy           # k-grid points
│   └── postprocessing.py
├── loop/               # Ploop component
│   ├── weights.npy
│   ├── inminmax.npy
│   ├── outminmax.npy
│   ├── nn_setup.json
│   ├── k.npy
│   └── postprocessing.py
├── ct/                 # Pct component
│   ├── weights.npy
│   ├── inminmax.npy
│   ├── outminmax.npy
│   ├── nn_setup.json
│   ├── k.npy
│   └── postprocessing.py
└── biascontraction.py  # Bias contraction matrix
```

## Data Storage Requirements

- **Per multipole**: ~50 MB
- **Full set (ℓ=0,2,4)**: ~150 MB
- **With all components**: ~200 MB total

## Offline Usage

For offline usage, ensure data is downloaded beforehand:

```python
import jaxeffort

# Download data while online
jaxeffort.force_update()

# Later, offline usage will work
P0 = jaxeffort.trained_emulators["pybird_mnuw0wacdm"]["0"]
```

## Environment Variables

You can customize the data directory using environment variables:

```bash
export JAXEFFORT_DATA_DIR=/custom/path/to/data
```

Then in Python:

```python
import os
os.environ['JAXEFFORT_DATA_DIR'] = '/custom/path/to/data'
import jaxeffort
```

## Troubleshooting

### Clear Cache

If you encounter issues, try clearing the cache:

```python
import shutil
import os

cache_dir = os.path.expanduser("~/.jaxeffort_data")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

# Re-import to trigger fresh download
import jaxeffort
jaxeffort.force_update()
```

### Verify Data Integrity

Check that all required files are present:

```python
import os
from pathlib import Path

def verify_emulator_data(path):
    """Verify emulator data structure."""
    required_files = {
        '11': ['weights.npy', 'inminmax.npy', 'outminmax.npy',
               'nn_setup.json', 'k.npy', 'postprocessing.py'],
        'loop': ['weights.npy', 'inminmax.npy', 'outminmax.npy',
                 'nn_setup.json', 'k.npy', 'postprocessing.py'],
        'ct': ['weights.npy', 'inminmax.npy', 'outminmax.npy',
              'nn_setup.json', 'k.npy', 'postprocessing.py']
    }

    path = Path(path)
    missing = []

    for component, files in required_files.items():
        comp_dir = path / component
        if not comp_dir.exists():
            missing.append(f"Directory: {component}/")
            continue

        for file in files:
            if not (comp_dir / file).exists():
                missing.append(f"File: {component}/{file}")

    if path / 'biascontraction.py':
        if not (path / 'biascontraction.py').exists():
            missing.append("File: biascontraction.py")

    if missing:
        print("Missing files:")
        for item in missing:
            print(f"  - {item}")
        return False
    else:
        print("✓ All required files present")
        return True

# Verify default emulator
verify_emulator_data("~/.jaxeffort_data/emulators/pybird_mnuw0wacdm/0")
```

## Manual Download

If automatic download fails, you can manually download the data:

1. Download from Zenodo: [DOI: 10.5281/zenodo.xxxxx]
2. Extract to `~/.jaxeffort_data/emulators/`
3. Verify structure matches expected format

## Data Sources

The pre-trained emulators are hosted on:
- **Zenodo**: Permanent DOI-based archive
- **GitHub Releases**: Alternative download location

## Citation

When using the pre-trained emulator data, please cite the data repository:

```bibtex
@dataset{jaxeffort_data,
  author = {Bonici, Marco and D'Amico, Guido and Bel, Julien and Carbone, Carmelita},
  title = {jaxeffort Pre-trained Emulator Data},
  year = {2025},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.xxxxx}
}
```