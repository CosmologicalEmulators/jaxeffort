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