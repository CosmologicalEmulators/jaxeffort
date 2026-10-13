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
P_ell = trained_emu.get_Pl(cosmo_params, bias_params, D)
```

Additionally, `jaxeffort` supports mapping theoretical spectra into observational spaces by modeling Alcock-Paczynski (AP) effects and Window Systematics Convolution:
```python3
from jaxeffort import apply_AP, window_convolution, q_par_perp
# Calculate mapping parameters from jaxace cosmology objects at z > 0.
q_par, q_perp = q_par_perp(z, cosmo_mcmc, cosmo_ref)

# Apply AP effects interpolating theoretical multipoles
# Historical PyBird artifacts store k_grid as (n_k, 1); AP takes a vector.
k_input = trained_emu.P11.k_grid.reshape(-1)
P0_ap, P2_ap, P4_ap = apply_AP(k_input, k_output, P0, P2, P4, q_par, q_perp)

# Convolve with Window Function matrix
P0_obs = window_convolution(Window_Matrix, P0_ap)
```

`apply_AP` supports both `(n_k,)` spectra and `(n_k, n_columns)` matrices.
Scalar AP parameters are shared by all columns, which is useful for projecting
bias Jacobians. For independent cosmologies/redshifts, either parameter can instead
be a vector of length `n_columns`; the columns are mapped with `jax.vmap`.
Projecting a bias Jacobian this way is exact for **Cubic** interpolation with
bias-independent AP parameters. Akima's interpolation weights depend on the
spectrum, so for Akima differentiate the complete prediction with JAX instead
of treating AP as a spectrum-independent linear operator.
Grids use the same **h-scaled wavenumber convention as Effort.jl**. Do not mix
physical-Mpc grids with the dimensionless-distance `q_par_perp` convention.

The Chebyshev-compressed window operators are JAX PyTrees, so prepared plans can
be passed directly to compiled functions, with their arrays remaining dynamic:

```python
import jax
from jaxeffort import prepare_ap_window_chebyshev, apply_AP_and_window

plan = prepare_ap_window_chebyshev(W0, W2, W4, k_dense, k_min, k_max, K=40)
project = jax.jit(apply_AP_and_window, static_argnames=("method", "n_GL_points"))
P0_obs, P2_obs, P4_obs = project(plan, k_input, P0, P2, P4, q_par, q_perp)
```

Choose the Chebyshev degree by comparing against dense AP followed by the same
window matrices; compression is an approximation, not an exact identity for
arbitrary spectra. `W0`, `W2`, and `W4` act separately on their respective input
multipoles. For a general multipole-mixing window, apply its full matrix to the
concatenated AP spectra using `window_convolution`.

For cosmology gradients, construct `w0waCDMCosmology` **inside**
the compiled function from its numerical arguments, then call `q_par_perp`.
The upstream cosmology dataclass is not itself a PyTree. `apply_AP_check` is an
eager SciPy reference, not a JIT/AD path. `apply_AP`, `window_convolution`,
`apply_chebyshev_operator`, and `apply_AP_and_window` support JIT and reverse-mode
differentiation; interpolation method and quadrature order are static choices.

`prepare_chebyshev_operator` and `prepare_ap_window_chebyshev` are host-side setup:
call them outside JIT with concrete domain bounds and an integer degree `K >= 1`.
Their upstream plan constructor does not support traced bounds or degree-zero
plans. The resulting plans can be passed as dynamic PyTree arguments to JIT-compiled
application functions; gradients can flow through their array data.

jaxace 0.7 is required. Older 0.5 releases produce NaN reverse-mode Akima
derivatives on flat/zero multipole columns; the upstream fix is exercised by
jaxeffort's projection tests.

`get_Pl` and `get_Pl_jacobian` forward additional keyword arguments to the
artifact's stochastic model. Numeric keyword values remain differentiable JAX
inputs. The stochastic function must accept the requested names.

The shipped artifacts match Effort.jl `reactant_dev` at `ea985a7`. Their
`multipole_setup.json` selects canonical bias, analytical Jacobian and stochastic
functions; `nn_setup.json` selects component postprocessing. Names may be flat or
nested under `emulator_description`, as in Julia. Unknown configured names raise
an error rather than silently falling back to stale artifact code. Legacy Python
files remain supported when no name is configured.

Both current velocileptors artifacts use bias order
`[b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4]`.
The new REPT artifact has 69 k points instead of the old 80. Do not reuse windows
or hard-coded grids from an older artifact without checking their coordinates.
Canonical bias/Jacobian pairs are validated against both Julia fixtures and JAX AD.

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
