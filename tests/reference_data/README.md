# Reference Data for Effort.jl Comparison Tests

This directory contains reference values from the Effort.jl implementation for comparison testing.

## Files

- `effort_jl_P0_ones.txt` - Monopole (l=0) reference values (complete: 74 values)
- `effort_jl_P2_ones.txt` - Quadrupole (l=2) reference values (complete: 74 values)
- `effort_jl_P4_ones.txt` - Hexadecapole (l=4) reference values (complete: 74 values)

## Input Configuration

All reference values were computed with:
- **Biases**: 11 values, all set to 1.0 (PyBird EFT bias parameters)
  - `[b1, b2, b3, b4, b5, b6, b7, f, cϵ0, cϵ1, cϵ2]`
- **Cosmology**: 9 values, all set to 1.0
  - `[z, ln10^10 As, ns, H0, omega_b, omega_c, Mnu, w0, wa]`
- **Growth factor D**: 1.0

## Generation

Reference data was generated using the `generate_reference_data.jl` script with Effort.jl
using the PyBirdmnuw0wacdm emulator (massive neutrinos, w0wa cosmology).

## Notes

All three multipole reference datasets are complete with 74 values each,
corresponding to the 74 k-values used in the PyBird emulator.

**Last updated**: 2025-10-24 with 11-parameter PyBird bias configuration

## AP projection references

`projection_cubic.txt` and `projection_akima.txt` contain every output point for
three independent columns and all three multipoles. Each column uses a different
AP parameter pair. The headers record Julia/Effort versions and the SHA256 of
the unmodified Julia projection source. `generate_projection_reference.jl`
specifies the complete inputs and refuses to overwrite existing fixtures.

Generated with Julia 1.12.6 and Effort 0.4.6 at commit `b6361bc`; its
`src/projection.jl` is identical to fetched `origin/main` (`19d3c99`).
`test_projection_jax.py` checks scalar and batched JAX paths against these text
fixtures unconditionally, and separately checks identity AP, exact quadrature
moments, JIT, vmap, and forward/reverse derivatives. CI does not require Julia.

## Canonical artifact and function references

`canonical_pybird_eftoflss.txt`, `canonical_velocileptors.txt`, and
`canonical_stochastic.txt` come directly from `src/builtins.jl` at Effort's
`reactant_dev` commit `ea985a7`. They fix the coefficient ordering and all
handwritten bias Jacobians independently of the Python implementation.

`canonical_spectra_{pybird,lpt,rept}.txt` contain k, spectrum, and all eleven
bias-Jacobian columns for each multipole, at a physical cosmology with mixed-sign
biases. The generator uses the existing Effort 0.4.6 evaluator, the canonical
Julia bias/Jacobian/stochastic functions, and the corrected Julia postprocessors
inside these verified archives:

| Model | Zenodo record | Archive SHA256 |
|---|---|---|
| PyBird | 20631625 | abc6969ebefde319f3e62f986ca37d4fcf2f4cca88e98d6a24a9ffcbc647055d |
| LPT | 20631626 | dffbb01c5383b393788190f2a16ebf79c763ec9ee31ff78bef08a7b473127690 |
| REPT | 20631359 | 277386435d225b068ef3c35669febee03a12910a961f4507b1a2a8740d0d5fef |

`generate_canonical_reference.jl` records the exact inputs and source hash,
preserves existing fixtures, and supports resuming missing outputs. The Python
tests check every element and require all three current emulator artifacts;
missing models are failures, not silent skips.
