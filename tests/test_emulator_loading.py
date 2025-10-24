"""Test emulator loading functionality."""

import pytest
import numpy as np
import jax.numpy as jnp
import tempfile
import shutil
import os
from pathlib import Path

from jaxeffort import (
    load_component_emulator,
    load_multipole_emulator,
    load_preprocessing,
    load_bias_combination,
)

from tests.fixtures.sample_cosmologies import (
    get_test_cosmology_array,
    get_test_biases,
)

from tests.fixtures.mock_emulator_data import (
    create_mock_emulator_directory,
)


class TestComponentEmulatorLoading:
    """Test loading individual component emulators."""

    def setup_method(self):
        """Create temporary directory with mock emulator data."""
        self.temp_dir = tempfile.mkdtemp(prefix="test_component_")
        self.mock_path = create_mock_emulator_directory(self.temp_dir)

    def teardown_method(self):
        """Clean up temporary directory."""
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_component_emulator(self):
        """Test loading a component emulator."""
        # Load P11 component
        p11_path = os.path.join(self.mock_path, "11")
        emulator = load_component_emulator(p11_path)

        # Check that emulator was created
        assert emulator is not None
        assert hasattr(emulator, 'k_grid')
        assert hasattr(emulator, 'in_MinMax')
        assert hasattr(emulator, 'out_MinMax')
        assert hasattr(emulator, 'postprocessing')

    def test_component_emulator_forward_pass(self):
        """Test forward pass through component emulator."""
        # Load emulator
        p11_path = os.path.join(self.mock_path, "11")
        emulator = load_component_emulator(p11_path)

        # Create test input
        test_input = get_test_cosmology_array()
        D = 1.0  # Growth factor

        # Run forward pass
        output = emulator.get_component(test_input, D)

        # Check output shape
        assert output.shape[0] == len(emulator.k_grid)
        assert output.ndim == 2
        assert np.all(np.isfinite(output))


class TestMultipoleEmulatorLoading:
    """Test loading multipole emulators."""

    @pytest.fixture
    def multipole_emu(self):
        """Get real multipole emulator."""
        import jaxeffort
        emulator = jaxeffort.trained_emulators.get('pybird_mnuw0wacdm', {}).get('0')
        if emulator is None:
            pytest.skip("Real emulator not available (not downloaded)")
        return emulator

    def test_load_multipole_emulator(self, multipole_emu):
        """Test loading a complete multipole emulator."""
        # Check that all components are loaded
        assert multipole_emu is not None
        assert multipole_emu.P11 is not None
        assert multipole_emu.Ploop is not None
        assert multipole_emu.Pct is not None
        assert multipole_emu.bias_combination is not None
        assert multipole_emu.stoch_model is not None

    def test_multipole_components_output(self, multipole_emu):
        """Test getting multipole components."""
        # Create test input
        cosmo = get_test_cosmology_array()
        D = 1.0  # Growth factor

        # Get components
        P11, Ploop, Pct = multipole_emu.get_multipole_components(cosmo, D)

        # Check outputs
        assert P11 is not None
        assert Ploop is not None
        assert Pct is not None

        # Check shapes are consistent (same number of k-points)
        assert P11.shape[0] == Ploop.shape[0]
        assert Ploop.shape[0] == Pct.shape[0]

        # Check that outputs are finite
        assert np.all(np.isfinite(P11))
        assert np.all(np.isfinite(Ploop))
        assert np.all(np.isfinite(Pct))

    def test_get_Pl_with_bias(self, multipole_emu):
        """Test getting P_ℓ with bias combination."""
        # Create test inputs
        cosmo = get_test_cosmology_array()
        biases = get_test_biases()
        D = 1.0

        # Get P_ℓ
        Pl = multipole_emu.get_Pl(cosmo, biases, D)

        # Check output
        assert Pl is not None
        assert Pl.ndim >= 1
        assert np.all(np.isfinite(Pl))

    def test_get_Pl_no_bias(self, multipole_emu):
        """Test getting raw P_ℓ components without bias."""
        # Create test inputs
        cosmo = get_test_cosmology_array()
        D = 1.0

        # Get raw components
        raw_components = multipole_emu.get_Pl_no_bias(cosmo, D)

        # Check output
        assert raw_components is not None
        assert raw_components.ndim == 2
        assert np.all(np.isfinite(raw_components))


class TestUtilityFunctions:
    """Test utility functions for loading emulator components."""

    def setup_method(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp(prefix="test_utils_")

    def teardown_method(self):
        """Clean up temporary directory."""
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_postprocessing(self):
        """Test loading postprocessing function."""
        # Create mock postprocessing file
        postproc_content = '''
def postprocessing(input_params, output, D, emulator):
    """Test postprocessing."""
    return output * 2.0
'''
        postproc_path = Path(self.temp_dir) / "postprocessing.py"
        postproc_path.write_text(postproc_content)

        # Load postprocessing
        postproc = load_preprocessing(self.temp_dir, "postprocessing")

        # Test function
        result = postproc(None, jnp.array([1.0, 2.0]), 1.0, None)
        np.testing.assert_allclose(result, jnp.array([2.0, 4.0]))

    def test_load_bias_combination(self):
        """Test loading bias combination function (supports legacy BiasContraction name)."""
        # Create mock bias combination file
        bias_content = '''
def BiasCombination(biases):
    """Test bias combination."""
    b1 = biases[0]
    return b1
'''
        bias_path = Path(self.temp_dir) / "biascombination.py"
        bias_path.write_text(bias_content)

        # Load bias combination
        bias_func = load_bias_combination(self.temp_dir)

        # Test function
        biases = jnp.array([2.0, 0.0, 0.0, 0.0])
        result = bias_func(biases)
        assert result == 2.0

    def test_missing_bias_combination_required(self):
        """Test that missing required bias combination raises error."""
        with pytest.raises(FileNotFoundError):
            load_bias_combination(self.temp_dir, required=True)

    def test_missing_bias_combination_optional(self):
        """Test that missing optional bias combination returns None."""
        result = load_bias_combination(self.temp_dir, required=False)
        assert result is None


class TestMLPMethods:
    """Test MLP class methods."""

    def test_mlp_get_component_method(self, tmp_path):
        """Test MLP.get_component method."""
        from jaxeffort import load_multipole_emulator
        from tests.fixtures.mock_emulator_data import create_mock_emulator_directory

        # Create mock emulator
        emulator_path = create_mock_emulator_directory(tmp_path)
        emulator = load_multipole_emulator(str(emulator_path))

        # Test get_component method
        x = jnp.ones(9)  # Mock emulator expects 9 inputs (as defined in mock_emulator_data)
        D = jnp.array(1.0)
        result = emulator.P11.get_component(x, D)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
