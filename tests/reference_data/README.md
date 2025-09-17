# Reference Data for Effort.jl Comparison Tests

This directory contains reference values from the Effort.jl implementation for comparison testing.

## Files

- `effort_jl_P0_ones.txt` - Monopole (l=0) reference values (complete: 74 values)
- `effort_jl_P2_ones.txt` - Quadrupole (l=2) reference values (complete: 74 values)
- `effort_jl_P4_ones.txt` - Hexadecapole (l=4) reference values (complete: 74 values)

## Input Configuration

All reference values were computed with:
- Biases: 8 values, all set to 1.0
- Cosmology: 9 values, all set to 1.0
- Growth factor D: 1.0

## Notes

All three multipole reference datasets are now complete with 74 values each,
corresponding to the 74 k-values used in the PyBird emulator.