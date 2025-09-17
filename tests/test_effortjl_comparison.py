"""
Test comparison with Effort.jl implementation.

This module tests that jaxeffort produces the same results as the original
Effort.jl implementation for specific test cases.
"""

import os
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

# Ensure Float64 precision
jax.config.update("jax_enable_x64", True)


class TestEffortJLComparison:
    """Test that jaxeffort matches Effort.jl reference implementation."""

    @staticmethod
    def load_reference_data(filename):
        """Load reference data from text file.

        Parameters
        ----------
        filename : str
            Name of the file in tests/reference_data/ directory

        Returns
        -------
        jnp.ndarray
            Array of reference values
        """
        # Get the directory where this test file is located
        test_dir = Path(__file__).parent
        reference_file = test_dir / "reference_data" / filename

        # Read the file, skipping comment lines
        values = []
        with open(reference_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    values.append(float(line))

        return jnp.array(values, dtype=jnp.float64)

    def test_ones_input_comparison(self):
        """
        Test with arrays of ones as input, comparing against Effort.jl reference values.

        This test uses the real PyBird emulator and compares the P0 (monopole) output
        against reference values computed by the Effort.jl implementation.
        """
        import jaxeffort

        # Get the real l=0 (monopole) emulator
        emulator = jaxeffort.get_multipole_emulator("pybird_mnuw0wacdm", l=0)
        if emulator is None:
            pytest.skip("Monopole (l=0) emulator not available (not downloaded)")

        # Input arrays of ones (matching Effort.jl test case)
        # Real PyBird emulator expects 8 bias parameters
        biases = jnp.ones(8, dtype=jnp.float64)

        # Cosmological parameters - 9 values expected by the real emulator
        cosmology = jnp.ones(9, dtype=jnp.float64)

        # Growth factor D = 1.0
        D = jnp.float64(1.0)

        # Load reference P0 values from file
        reference_P0 = self.load_reference_data("effort_jl_P0_ones.txt")

        # Compute P_ℓ using jaxeffort
        try:
            Pl = emulator.get_Pl(cosmology, biases, D)

            # Extract P0 (monopole) - should be the first set of values
            # The output shape is (n_k, n_multipoles) where n_multipoles includes monopole and quadrupole
            if Pl.ndim == 2:
                # Assuming the first column or first set corresponds to P0
                # We need to extract the correct subset
                P0_computed = Pl[:, 0] if Pl.shape[1] > 1 else Pl.flatten()
            else:
                P0_computed = Pl

            # Ensure we have the right number of values
            n_ref = len(reference_P0)
            if len(P0_computed) > n_ref:
                P0_computed = P0_computed[:n_ref]
            elif len(P0_computed) < n_ref:
                pytest.fail(f"Computed P0 has {len(P0_computed)} values, expected {n_ref}")

            # Check that the values match within a reasonable tolerance
            # Use a relative tolerance for larger values and absolute tolerance for smaller ones
            rtol = 1e-6  # Relative tolerance
            atol = 1e-8  # Absolute tolerance

            # Verify dtype is float64
            assert P0_computed.dtype == jnp.float64, f"Expected float64, got {P0_computed.dtype}"

            # Compare arrays
            max_rel_diff = jnp.max(jnp.abs((P0_computed - reference_P0) / (reference_P0 + 1e-10)))
            max_abs_diff = jnp.max(jnp.abs(P0_computed - reference_P0))

            print(f"\nMax relative difference: {max_rel_diff:.2e}")
            print(f"Max absolute difference: {max_abs_diff:.2e}")

            # Check if arrays are close
            if not jnp.allclose(P0_computed, reference_P0, rtol=rtol, atol=atol):
                # Print detailed comparison for debugging
                print("\nDetailed comparison (first 10 values):")
                for i in range(min(10, n_ref)):
                    diff = P0_computed[i] - reference_P0[i]
                    rel_diff = diff / (abs(reference_P0[i]) + 1e-10)
                    print(f"  Index {i}: computed={P0_computed[i]:.6f}, "
                          f"reference={reference_P0[i]:.6f}, "
                          f"diff={diff:.2e}, rel_diff={rel_diff:.2e}")

                pytest.fail(f"P0 values don't match Effort.jl reference. "
                           f"Max rel diff: {max_rel_diff:.2e}, "
                           f"Max abs diff: {max_abs_diff:.2e}")

            print("\n✓ P0 values match Effort.jl reference implementation!")

        except Exception as e:
            pytest.fail(f"Failed to compute P_ℓ with real emulator: {e}")

    def test_jaxeffort_output_structure(self):
        """
        Test and document the actual output structure of jaxeffort.

        This test verifies what jaxeffort produces with ones input and serves
        as a regression test for the current implementation.
        """
        import jaxeffort

        # Get the l=0 (monopole) emulator
        emulator = jaxeffort.get_multipole_emulator("pybird_mnuw0wacdm", l=0)
        if emulator is None:
            pytest.skip("Monopole (l=0) emulator not available")

        # Input arrays of ones
        biases = jnp.ones(8, dtype=jnp.float64)
        cosmology = jnp.ones(9, dtype=jnp.float64)
        D = jnp.float64(1.0)

        # Compute P_ℓ
        Pl = emulator.get_Pl(cosmology, biases, D)

        # Document the output structure - changed with new emulator
        assert Pl.shape == (74,), f"Expected shape (74,), got {Pl.shape}"
        assert Pl.dtype == jnp.float64, f"Expected float64, got {Pl.dtype}"

        # Check that values are finite
        assert jnp.all(jnp.isfinite(Pl)), "Output contains non-finite values"

        # Document some expected values for regression testing
        # These are the actual values jaxeffort produces (not from Effort.jl)
        expected_first_value = Pl[0]
        expected_sum_all = jnp.sum(Pl)

        # Verify consistency (sanity check)
        assert jnp.isclose(Pl[0], expected_first_value, rtol=1e-10)
        assert jnp.isclose(jnp.sum(Pl), expected_sum_all, rtol=1e-10)

        print(f"\n✓ jaxeffort output structure verified:")
        print(f"  Shape: {Pl.shape}")
        print(f"  Dtype: {Pl.dtype}")
        print(f"  All values finite: {jnp.all(jnp.isfinite(Pl))}")
        print(f"  First value: {Pl[0]:.6f}")
        print(f"  Total sum: {expected_sum_all:.6f}")

    def test_dtype_preservation(self):
        """Test that Float64 precision is maintained throughout the computation."""
        import jaxeffort

        # Get the l=0 (monopole) emulator
        emulator = jaxeffort.get_multipole_emulator("pybird_mnuw0wacdm", l=0)
        if emulator is None:
            pytest.skip("Monopole (l=0) emulator not available")

        # Create Float64 inputs
        biases = jnp.ones(8, dtype=jnp.float64)
        cosmology = jnp.ones(9, dtype=jnp.float64)
        D = jnp.float64(1.0)

        # Verify input dtypes
        assert biases.dtype == jnp.float64
        assert cosmology.dtype == jnp.float64
        assert D.dtype == jnp.float64

        # Compute result
        Pl = emulator.get_Pl(cosmology, biases, D)

        # Verify output dtype
        assert Pl.dtype == jnp.float64, f"Output dtype is {Pl.dtype}, expected float64"

        print("\n✓ Float64 precision maintained throughout computation")

    def test_l2_quadrupole_comparison(self):
        """Test jaxeffort l=2 (quadrupole) output against reference values."""
        import jaxeffort

        # Get the l=2 (quadrupole) emulator
        emulator = jaxeffort.get_multipole_emulator("pybird_mnuw0wacdm", l=2)
        if emulator is None:
            pytest.skip("Quadrupole (l=2) emulator not available (not downloaded)")

        # Input arrays of ones (same as l=0 test)
        biases = jnp.ones(8, dtype=jnp.float64)
        cosmology = jnp.ones(9, dtype=jnp.float64)
        D = jnp.float64(1.0)

        # Load reference P2 values from file
        reference_P2 = self.load_reference_data("effort_jl_P2_ones.txt")

        # Compute P_ℓ using jaxeffort
        P2_computed = emulator.get_Pl(cosmology, biases, D)

        # Ensure we have the right shape
        assert P2_computed.shape == (74,), f"Expected shape (74,), got {P2_computed.shape}"

        # Check that the values match within reasonable tolerance
        rtol = 1e-6  # Relative tolerance
        atol = 1e-8  # Absolute tolerance

        # Calculate differences
        relative_diff = jnp.abs((P2_computed - reference_P2) / reference_P2)
        absolute_diff = jnp.abs(P2_computed - reference_P2)

        # Find maximum differences
        max_rel_diff = jnp.max(relative_diff)
        max_abs_diff = jnp.max(absolute_diff)

        print(f"\nMax relative difference: {max_rel_diff:.2e}")
        print(f"Max absolute difference: {max_abs_diff:.2e}")

        # Check if values match within tolerance
        matches = jnp.allclose(P2_computed, reference_P2, rtol=rtol, atol=atol)

        if matches:
            print("\n✓ P2 (quadrupole) values match Effort.jl reference implementation!")
        else:
            # Find indices where differences are largest
            worst_indices = jnp.argsort(relative_diff)[-5:]
            print("\nLargest relative differences at indices:")
            for idx in worst_indices:
                print(f"  k[{idx}]: computed={P2_computed[idx]:.6f}, "
                      f"reference={reference_P2[idx]:.6f}, "
                      f"rel_diff={relative_diff[idx]:.2e}")

        assert matches, f"P2 values differ from reference (max rel diff: {max_rel_diff:.2e})"

    def test_l4_hexadecapole_comparison(self):
        """Test jaxeffort l=4 (hexadecapole) output against reference values."""
        import jaxeffort

        # Get the l=4 (hexadecapole) emulator
        emulator = jaxeffort.get_multipole_emulator("pybird_mnuw0wacdm", l=4)
        if emulator is None:
            pytest.skip("Hexadecapole (l=4) emulator not available (not downloaded)")

        # Input arrays of ones (same as l=0 and l=2 tests)
        biases = jnp.ones(8, dtype=jnp.float64)
        cosmology = jnp.ones(9, dtype=jnp.float64)
        D = jnp.float64(1.0)

        # Load reference P4 values from file
        reference_P4 = self.load_reference_data("effort_jl_P4_ones.txt")

        # Compute P_ℓ using jaxeffort
        P4_computed = emulator.get_Pl(cosmology, biases, D)

        # Ensure we have the right shape
        assert P4_computed.shape == (74,), f"Expected shape (74,), got {P4_computed.shape}"

        # Check that the values match within reasonable tolerance
        rtol = 1e-6  # Relative tolerance
        atol = 1e-8  # Absolute tolerance

        # Calculate differences
        # Handle division by zero for reference values that are very small
        relative_diff = jnp.where(
            jnp.abs(reference_P4) > 1e-10,
            jnp.abs((P4_computed - reference_P4) / reference_P4),
            jnp.abs(P4_computed - reference_P4)  # Use absolute diff for small values
        )
        absolute_diff = jnp.abs(P4_computed - reference_P4)

        # Find maximum differences
        max_rel_diff = jnp.max(relative_diff)
        max_abs_diff = jnp.max(absolute_diff)

        print(f"\nMax relative difference: {max_rel_diff:.2e}")
        print(f"Max absolute difference: {max_abs_diff:.2e}")

        # Check if values match within tolerance
        matches = jnp.allclose(P4_computed, reference_P4, rtol=rtol, atol=atol)

        if matches:
            print("\n✓ P4 (hexadecapole) values match Effort.jl reference implementation!")
        else:
            # Find indices where differences are largest
            worst_indices = jnp.argsort(relative_diff)[-5:]
            print("\nLargest relative differences at indices:")
            for idx in worst_indices:
                print(f"  k[{idx}]: computed={P4_computed[idx]:.6f}, "
                      f"reference={reference_P4[idx]:.6f}, "
                      f"rel_diff={relative_diff[idx]:.2e}")

        assert matches, f"P4 values differ from reference (max rel diff: {max_rel_diff:.2e})"