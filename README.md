# jaxeffort
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://drive.google.com/file/d/1FxnXViYqIAg8Vu13Tippf0m7RIuSseEu/view?usp=sharing)
[![codecov](https://codecov.io/gh/CosmologicalEmulators/jaxeffort/branch/develop/graph/badge.svg?token=GPLEGMIU08)](https://codecov.io/gh/CosmologicalEmulators/jaxeffort)
[![arXiv](https://img.shields.io/badge/arXiv-2501.04639-b31b1b.svg)](https://arxiv.org/abs/2501.04639)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://cosmologicalemulators.github.io/jaxeffort/stable/)
[![Documentation Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cosmologicalemulators.github.io/jaxeffort/dev/)

JAX-based emulator for galaxy power spectra with bias modeling and EFT corrections.

## Documentation

- **[Stable Documentation](https://cosmologicalemulators.github.io/jaxeffort/stable/)** - Latest release documentation
- **[Development Documentation](https://cosmologicalemulators.github.io/jaxeffort/dev/)** - Latest development version documentation

## Installation and usage

In order to install `jaxeffort`, you can just run

```bash
pip install jaxeffort
```

If you prefer to use the latest version from the repository, you can clone it, enter it, and run

```bash
pip install .
```

In order to use the emulators, you have to import `jaxeffort` and load a trained emulator

```python3
import jaxeffort
import jax.numpy as np
trained_emu = jaxeffort.load_multipole_emulator("/path/to/emu/")
```
Then you are good to go! You have to create input arrays for cosmological and bias parameters and retrieve your calculation result

```python3
cosmo_params = np.array([...])  # cosmological parameters
bias_params = np.array([...])   # bias parameters
result = trained_emu.get_Pl(cosmo_params, bias_params, D)
```

For a more detailed explanation, check the tutorial in the `notebooks` folder, which also shows a comparison with standard power spectrum calculations.

## Citing

Free usage of the software in this repository is provided, given that you cite our release paper.

M. Bonici, G. D'Amico, J. Bel, C. Carbone, [_Effort.jl: a fast and differentiable emulator for the Effective Field Theory of the Large Scale Structure of the Universe_](https://dx.doi.org/10.1088/1475-7516/2025/09/044), JCAP 09 (2025) 044

```bibtex
@article{Bonici_2025,
  doi = {10.1088/1475-7516/2025/09/044},
  url = {https://dx.doi.org/10.1088/1475-7516/2025/09/044},
  year = {2025},
  month = {sep},
  publisher = {IOP Publishing},
  volume = {2025},
  number = {09},
  pages = {044},
  author = {Bonici, Marco and D'Amico, Guido and Bel, Julien and Carbone, Carmelita},
  title = {Effort.jl: a fast and differentiable emulator for the Effective Field Theory of the Large Scale Structure of the Universe},
  journal = {Journal of Cosmology and Astroparticle Physics}
}
```
