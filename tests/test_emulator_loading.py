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
    load_multipole_noise_emulator,
    load_preprocessing,
    load_bias_contraction,
)

from tests.fixtures.mock_emulator_data import (
    create_mock_emulator_directory,
    create_mock_noise_emulator_directory,
)

from tests.fixtures.sample_cosmologies import (
    get_test_cosmology_array,
    get_test_biases,
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
        assert hasattr(emulator, 'NN_params')
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

    def setup_method(self):
        """Create temporary directory with mock emulator data."""
        self.temp_dir = tempfile.mkdtemp(prefix="test_multipole_")
        self.mock_path = create_mock_emulator_directory(self.temp_dir)

    def teardown_method(self):
        """Clean up temporary directory."""
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_multipole_emulator(self):
        """Test loading a complete multipole emulator."""
        # Load multipole emulator
        multipole_emu = load_multipole_emulator(self.mock_path)

        # Check that all components are loaded
        assert multipole_emu is not None
        assert multipole_emu.P11 is not None
        assert multipole_emu.Ploop is not None
        assert multipole_emu.Pct is not None
        assert multipole_emu.bias_contraction is not None

    def test_multipole_components_output(self):
        """Test getting multipole components."""
        # Load emulator
        multipole_emu = load_multipole_emulator(self.mock_path)

        # Create test input
        cosmo = get_test_cosmology_array()
        D = 1.0  # Growth factor

        # Get components
        P11, Ploop, Pct = multipole_emu.get_multipole_components(cosmo, D)

        # Check outputs
        assert P11 is not None
        assert Ploop is not None
        assert Pct is not None

        # Check shapes are consistent
        assert P11.shape == Ploop.shape
        assert Ploop.shape == Pct.shape

        # Check that outputs are finite
        assert np.all(np.isfinite(P11))
        assert np.all(np.isfinite(Ploop))
        assert np.all(np.isfinite(Pct))

    def test_get_Pl_with_bias(self):
        """Test getting P_ℓ with bias contraction."""
        # Load emulator
        multipole_emu = load_multipole_emulator(self.mock_path)

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

    def test_get_Pl_no_bias(self):
        """Test getting raw P_ℓ components without bias."""
        # Load emulator
        multipole_emu = load_multipole_emulator(self.mock_path)

        # Create test inputs
        cosmo = get_test_cosmology_array()
        D = 1.0

        # Get raw components
        raw_components = multipole_emu.get_Pl_no_bias(cosmo, D)

        # Check output
        assert raw_components is not None
        assert raw_components.ndim == 2
        assert np.all(np.isfinite(raw_components))


class TestMultipoleNoiseEmulatorLoading:
    """Test loading multipole emulators with noise component."""

    def setup_method(self):
        """Create temporary directory with mock emulator data."""
        self.temp_dir = tempfile.mkdtemp(prefix="test_noise_")
        self.mock_path = create_mock_noise_emulator_directory(self.temp_dir)

    def teardown_method(self):
        """Clean up temporary directory."""
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_multipole_noise_emulator(self):
        """Test loading a multipole emulator with noise."""
        # Load emulator
        noise_emu = load_multipole_noise_emulator(self.mock_path)

        # Check all components
        assert noise_emu is not None
        assert noise_emu.multipole_emulator is not None
        assert noise_emu.noise_emulator is not None
        assert noise_emu.bias_contraction is not None

        # Check multipole components
        assert noise_emu.multipole_emulator.P11 is not None
        assert noise_emu.multipole_emulator.Ploop is not None
        assert noise_emu.multipole_emulator.Pct is not None

    def test_noise_emulator_get_Pl(self):
        """Test getting P_ℓ with noise component."""
        # Load emulator
        noise_emu = load_multipole_noise_emulator(self.mock_path)

        # Create test inputs
        cosmo = get_test_cosmology_array()
        biases = get_test_biases()
        D = 1.0

        # Get P_ℓ with noise
        Pl = noise_emu.get_Pl(cosmo, biases, D)

        # Check output
        assert Pl is not None
        assert np.all(np.isfinite(Pl))


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

    def test_load_bias_contraction(self):
        """Test loading bias contraction function."""
        # Create mock bias contraction file
        bias_content = '''
def BiasContraction(biases, stacked_array):
    """Test bias contraction."""
    b1 = biases[0]
    return stacked_array * b1
'''
        bias_path = Path(self.temp_dir) / "biascontraction.py"
        bias_path.write_text(bias_content)

        # Load bias contraction
        bias_func = load_bias_contraction(self.temp_dir)

        # Test function
        biases = jnp.array([2.0, 0.0, 0.0, 0.0])
        data = jnp.array([1.0, 2.0, 3.0])
        result = bias_func(biases, data)
        np.testing.assert_allclose(result, jnp.array([2.0, 4.0, 6.0]))

    def test_missing_bias_contraction_required(self):
        """Test that missing required bias contraction raises error."""
        with pytest.raises(FileNotFoundError):
            load_bias_contraction(self.temp_dir, required=True)

    def test_missing_bias_contraction_optional(self):
        """Test that missing optional bias contraction returns None."""
        result = load_bias_contraction(self.temp_dir, required=False)
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

class TestBackwardCompatibility:
    """Test backward compatibility methods."""

    def test_mlp_apply_method(self, tmp_path):
        """Test MLP.apply method for backward compatibility."""
        from jaxeffort import load_multipole_emulator
        from tests.fixtures.mock_emulator_data import create_mock_emulator_directory

        # Create mock emulator
        emulator_path = create_mock_emulator_directory(tmp_path)
        emulator = load_multipole_emulator(str(emulator_path))

        # Test apply method (backward compatibility)
        x = jnp.ones(8)  # Mock emulator expects 8 inputs
        params = {}  # Not used but required for signature
        result = emulator.P11.apply(params, x)
        assert result is not None


class TestBiasContractionLoading:
    """Test different bias contraction loading scenarios."""

    def test_load_bias_contraction_lowercase(self, tmp_path):
        """Test loading bias contraction with lowercase name."""
        from jaxeffort.jaxeffort import load_bias_contraction

        # Create a file with lowercase biascontraction
        bc_file = tmp_path / "biascontraction.py"
        bc_file.write_text("""def biascontraction(biases, array):
    return biases @ array
""")

        # This should load the lowercase version
        bc_func = load_bias_contraction(str(tmp_path), required=True)
        assert bc_func is not None
        assert callable(bc_func)

    def test_load_bias_contraction_missing_optional(self, tmp_path):
        """Test loading missing bias contraction when optional."""
        from jaxeffort.jaxeffort import load_bias_contraction

        # Don't create the file - test missing file case
        # This should return None when optional
        bc_func = load_bias_contraction(str(tmp_path), required=False)
        assert bc_func is None

    def test_missing_bias_contraction_error(self):
        """Test error when bias contraction is missing but required."""
        from jaxeffort.jaxeffort import MultipoleEmulators, MLP
        from unittest.mock import MagicMock

        # Create mock MLPs
        mock_p11 = MagicMock(spec=MLP)
        mock_ploop = MagicMock(spec=MLP)
        mock_pct = MagicMock(spec=MLP)

        # Set up mock return values with proper shape
        mock_p11.get_component.return_value = jnp.ones((74, 6))
        mock_ploop.get_component.return_value = jnp.ones((74, 6))
        mock_pct.get_component.return_value = jnp.ones((74, 6))

        # Create MultipoleEmulators without bias contraction
        emulator = MultipoleEmulators(
            P11=mock_p11,
            Ploop=mock_ploop,
            Pct=mock_pct,
            bias_contraction=None  # Missing bias contraction
        )

        # This should raise an error when trying to compute with biases
        cosmology = jnp.ones(9)
        biases = jnp.ones(4)
        D = 1.0

        with pytest.raises(ValueError, match="biascontraction is required"):
            emulator.get_Pl(cosmology, biases, D)
