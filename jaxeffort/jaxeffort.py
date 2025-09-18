from typing import Sequence, List, Tuple, Dict, Any
import os
import json
import importlib.util

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

# Import all background cosmology functions from jaxace
# Handle different jaxace versions that may export Ωma or Ωm_a
from jaxace.background import (
    W0WaCDMCosmology,
    a_z, E_a, E_z, dlogEdloga, Ωm_a,
    D_z, f_z, D_f_z,
    r_z, dA_z, dL_z,
    ρc_z, Ωtot_z,
    # Neutrino functions
    F, dFdy, ΩνE2,
    # Growth solver
    growth_solver, growth_ode_system
)

# Import neural network infrastructure from jaxace
from jaxace import (
    init_emulator,
    FlaxEmulator,
    maximin,
    inv_maximin
)

jax.config.update("jax_enable_x64", True)


class MLP:
    """
    Effort MLP emulator using jaxace infrastructure.

    This class wraps a jaxace FlaxEmulator with Effort-specific functionality
    for galaxy power spectrum computation.
    """

    def __init__(self,
                 emulator: FlaxEmulator,
                 k_grid: np.ndarray,
                 in_MinMax: np.ndarray,
                 out_MinMax: np.ndarray,
                 postprocessing: callable,
                 emulator_description: Dict[str, Any],
                 nn_dict: Dict[str, Any]):
        """
        Initialize MLP with jaxace emulator and Effort-specific components.

        Args:
            emulator: jaxace FlaxEmulator instance
            k_grid: k-space grid for power spectrum
            in_MinMax: Input normalization parameters
            out_MinMax: Output normalization parameters
            postprocessing: Postprocessing function
            emulator_description: Emulator metadata
            nn_dict: Neural network configuration dictionary
        """
        self.emulator = emulator
        self.k_grid = jnp.asarray(k_grid)
        self.in_MinMax = jnp.asarray(in_MinMax)
        self.out_MinMax = jnp.asarray(out_MinMax)
        self.postprocessing = postprocessing
        self.emulator_description = emulator_description

    def maximin(self, input):
        """Normalize input using jaxace's maximin function."""
        return maximin(input, self.in_MinMax)

    def inv_maximin(self, output):
        """Denormalize output using jaxace's inv_maximin function."""
        return inv_maximin(output, self.out_MinMax)

    def get_component(self, input, D):
        """
        Get raw component output without bias contraction, matching Effort.jl's get_component.

        This method delegates to a JIT-compiled implementation.
        """
        # Check postprocessing signature once and create appropriate JIT function
        if not hasattr(self, '_jit_get_component'):
            import inspect
            try:
                sig = inspect.signature(self.postprocessing)
                num_params = len(sig.parameters)
            except (ValueError, TypeError):
                # Fallback for cases where inspect.signature fails (e.g., some lambdas in Python 3.10)
                # Try calling with different signatures to determine the correct one
                try:
                    # Try with 4 parameters (test version)
                    test_result = self.postprocessing(jnp.ones(1), jnp.ones(1), 1.0, self)
                    num_params = 4
                except (TypeError, ValueError):
                    # Must be 3 parameters (production version)
                    num_params = 3

            if num_params == 4:
                # Test version with emulator parameter
                @partial(jax.jit, static_argnums=(0,))
                def _jit_get_component_with_emulator(self, input, D):
                    norm_input = self.maximin(input)
                    norm_model_output = self.emulator.run_emulator(norm_input)
                    model_output = self.inv_maximin(norm_model_output)
                    processed_model_output = self.postprocessing(input, model_output, D, self)
                    reshaped_output = processed_model_output.reshape(
                        (len(self.k_grid), int(len(processed_model_output) / len(self.k_grid))), order="F"
                    )
                    return reshaped_output
                self._jit_get_component = _jit_get_component_with_emulator
            else:
                # Production version without emulator parameter
                @partial(jax.jit, static_argnums=(0,))
                def _jit_get_component_standard(self, input, D):
                    norm_input = self.maximin(input)
                    norm_model_output = self.emulator.run_emulator(norm_input)
                    model_output = self.inv_maximin(norm_model_output)
                    processed_model_output = self.postprocessing(input, model_output, D)
                    reshaped_output = processed_model_output.reshape(
                        (len(self.k_grid), int(len(processed_model_output) / len(self.k_grid))), order="F"
                    )
                    return reshaped_output
                self._jit_get_component = _jit_get_component_standard

        return self._jit_get_component(self, input, D)


class MultipoleEmulators:
    def __init__(self, P11: MLP, Ploop: MLP, Pct: MLP, bias_contraction: callable):
        """
        Initializes the MultipoleEmulators class with three MLP instances and bias contraction.

        Args:
            P11 (MLP): MLP instance for P11 emulator.
            Ploop (MLP): MLP instance for Ploop emulator.
            Pct (MLP): MLP instance for Pct emulator.
            bias_contraction (callable): Bias contraction function for the multipole.
        """
        self.P11 = P11
        self.Ploop = Ploop
        self.Pct = Pct
        self.bias_contraction = bias_contraction

    @partial(jax.jit, static_argnums=(0,))
    def get_multipole_components(self, inputs: np.array, D) -> Tuple[np.array, np.array, np.array]:
        """
        Computes the raw component outputs for all three emulators given an input array.

        This method is JIT-compiled for performance.

        Args:
            inputs (np.array): Input data to the emulators.
            D: Growth factor.

        Returns:
            Tuple[np.array, np.array, np.array]: Component outputs of P11, Ploop, and Pct emulators.
        """
        P11_output = self.P11.get_component(inputs, D)
        Ploop_output = self.Ploop.get_component(inputs, D)
        Pct_output = self.Pct.get_component(inputs, D)

        return P11_output, Ploop_output, Pct_output

    def get_Pl(self, cosmology, biases, D):
        """
        Get P_ℓ using the multipole's bias contraction function.
        Matches Effort.jl where BiasContraction is at PℓEmulator level only.

        This method uses JIT compilation for performance.
        """
        if self.bias_contraction is None:
            raise ValueError("biascontraction is required to compute P_ℓ with biases")

        # Create JIT-compiled version on first call
        if not hasattr(self, '_jit_get_Pl'):
            @partial(jax.jit, static_argnums=(0,))
            def _jit_get_Pl(self, cosmology, biases, D):
                P11_comp, Ploop_comp, Pct_comp = self.get_multipole_components(cosmology, D)
                stacked_array = jnp.hstack((P11_comp, Ploop_comp, Pct_comp))
                return self.bias_contraction(biases, stacked_array)
            self._jit_get_Pl = _jit_get_Pl

        return self._jit_get_Pl(self, cosmology, biases, D)

    def get_Pl_no_bias(self, cosmology, D):
        """Get raw components without bias contraction."""
        P11_output, Ploop_output, Pct_output = self.get_multipole_components(cosmology, D)
        return jnp.hstack((P11_output, Ploop_output, Pct_output))


class MultipoleNoiseEmulator:
    def __init__(self, multipole_emulator: MultipoleEmulators, noise_emulator: MLP, bias_contraction: callable):
        """
        Initializes the MultipoleNoiseEmulator with a multipole emulator and a noise emulator.

        Args:
            multipole_emulator (MultipoleEmulators): An instance of the MultipoleEmulators class.
            noise_emulator (MLP): An instance of the MLP class representing the noise emulator.
            bias_contraction (callable): Overall bias contraction function.
        """
        self.multipole_emulator = multipole_emulator
        self.noise_emulator = noise_emulator
        self.bias_contraction = bias_contraction

    def get_Pl(self, cosmology, biases, D):
        """
        Get P_ℓ with noise, using bias contraction.

        This method uses JIT compilation for performance.
        """
        # Create JIT-compiled version on first call
        if not hasattr(self, '_jit_get_Pl'):
            @partial(jax.jit, static_argnums=(0,))
            def _jit_get_Pl(self, cosmology, biases, D):
                # Get all components
                P11_comp, Ploop_comp, Pct_comp = self.multipole_emulator.get_multipole_components(cosmology, D)
                Noise_comp = self.noise_emulator.get_component(cosmology, D)
                stacked_array = jnp.hstack((P11_comp, Ploop_comp, Pct_comp, Noise_comp))
                # Use the overall bias contraction
                return self.bias_contraction(biases, stacked_array)
            self._jit_get_Pl = _jit_get_Pl

        return self._jit_get_Pl(self, cosmology, biases, D)

    def get_Pl_no_bias(self, cosmology, D):
        """Get raw components without bias contraction."""
        P11_output, Ploop_output, Pct_output = self.multipole_emulator.get_multipole_components(cosmology, D)
        Noise_output = self.noise_emulator.get_component(cosmology, D)
        return jnp.hstack((P11_output, Ploop_output, Pct_output, Noise_output))


def load_preprocessing(root_path, filename):
    """Load postprocessing function from Python file."""
    spec = importlib.util.spec_from_file_location(filename, root_path + "/" + filename + ".py")
    test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test)
    return test.postprocessing


def load_bias_contraction(root_path, filename="biascontraction", required=True):
    """Load bias contraction function from Python file."""
    filepath = root_path + "/" + filename + ".py"
    import os
    if not os.path.exists(filepath):
        if required:
            raise FileNotFoundError(f"Bias contraction file not found: {filepath}")
        return None

    spec = importlib.util.spec_from_file_location(filename, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Try to get BiasContraction function (capital B) or biascontraction (lowercase)
    if hasattr(module, 'BiasContraction'):
        return module.BiasContraction
    elif hasattr(module, 'biascontraction'):
        return module.biascontraction
    else:
        if required:
            raise AttributeError(f"No BiasContraction or biascontraction function found in {filepath}")
        return None


def load_component_emulator(folder_path):
    """Load a component emulator (P11, Ploop, Pct, or Noise) using jaxace infrastructure."""
    from pathlib import Path
    folder_path = Path(folder_path)

    # Load normalization parameters
    in_MinMax = jnp.load(folder_path / "inminmax.npy")
    out_MinMax = jnp.load(folder_path / "outminmax.npy")

    # Load neural network configuration
    with open(folder_path / "nn_setup.json", 'r') as f:
        nn_dict = json.load(f)

    # Load k-grid and weights
    k_grid = jnp.load(folder_path / "k.npy")
    weights = jnp.load(folder_path / "weights.npy")

    # Initialize jaxace emulator
    jaxace_emulator = init_emulator(
        nn_dict=nn_dict,
        weight=weights,
        validate=True
    )

    # Load postprocessing
    postprocessing = load_preprocessing(str(folder_path), "postprocessing")

    # Extract emulator description
    emulator_description = nn_dict.get("emulator_description", {})

    # Create MLP instance with jaxace backend
    return MLP(
        emulator=jaxace_emulator,
        k_grid=k_grid,
        in_MinMax=in_MinMax,
        out_MinMax=out_MinMax,
        postprocessing=postprocessing,
        emulator_description=emulator_description,
        nn_dict=nn_dict
    )


def load_multipole_emulator(folder_path: str) -> MultipoleEmulators:
    """
    Loads the three multipole emulators (P11, Ploop, Pct) from their respective subfolders.
    Bias contraction is loaded at the multipole level, matching Effort.jl structure.

    Args:
        folder_path (str): The path to the folder containing the subfolders `11`, `loop`, and `ct`.

    Returns:
        MultipoleEmulators: An instance of the MultipoleEmulators class containing the loaded emulators.
    """
    from pathlib import Path
    folder_path = Path(folder_path)

    # Define subfolder paths
    P11_path = folder_path / "11"
    Ploop_path = folder_path / "loop"
    Pct_path = folder_path / "ct"

    # Load each component emulator (no bias contraction at component level)
    P11_emulator = load_component_emulator(P11_path)
    Ploop_emulator = load_component_emulator(Ploop_path)
    Pct_emulator = load_component_emulator(Pct_path)

    # Load multipole-level bias contraction - this is required (matches Effort.jl PℓEmulator)
    multipole_bias_contraction = load_bias_contraction(str(folder_path), required=True)

    # Return the MultipoleEmulators instance with bias contraction
    return MultipoleEmulators(P11_emulator, Ploop_emulator, Pct_emulator, multipole_bias_contraction)


def get_stoch_terms(cϵ0, cϵ1, cϵ2, n_bar, k_grid, k_nl=0.7):
    """
    Compute stochastic power spectrum terms.

    Implements the stochastic contributions to the power spectrum
    following the Effective Field Theory (EFT) approach.

    Args:
        cϵ0: Constant stochastic parameter (epsilon_0)
        cϵ1: k^2-dependent stochastic parameter (epsilon_1)
        cϵ2: k^2-dependent stochastic parameter for quadrupole (epsilon_2)
        n_bar: Mean number density of galaxies
        k_grid: Array of k values
        k_nl: Non-linear scale (default: 0.7)

    Returns:
        Tuple of (P_stoch_0, P_stoch_2) for monopole and quadrupole
    """
    k_grid = jnp.asarray(k_grid)
    P_stoch_0 = (1 / n_bar) * (cϵ0 + cϵ1 * (k_grid / k_nl)**2)
    P_stoch_2 = (1 / n_bar) * (cϵ2 * (k_grid / k_nl)**2)
    return P_stoch_0, P_stoch_2


def load_multipole_noise_emulator(folder_path: str) -> MultipoleNoiseEmulator:
    """
    Loads the multipole noise emulator, including a multipole emulator and a noise emulator.
    Expects bias contraction at the top level for the combined emulator.

    Args:
        folder_path (str): The path to the folder containing the trained emulators.
                           The folder should contain subfolders for the multipole emulator and a 'st' subfolder for the noise emulator.

    Returns:
        MultipoleNoiseEmulator: An instance of the MultipoleNoiseEmulator class.
    """
    from pathlib import Path
    folder_path = Path(folder_path)

    # Define subfolder paths
    P11_path = folder_path / "11"
    Ploop_path = folder_path / "loop"
    Pct_path = folder_path / "ct"
    noise_path = folder_path / "st"

    # Load component emulators (no bias contraction at component level)
    P11_emulator = load_component_emulator(P11_path)
    Ploop_emulator = load_component_emulator(Ploop_path)
    Pct_emulator = load_component_emulator(Pct_path)
    noise_emulator = load_component_emulator(noise_path)

    # For MultipoleNoiseEmulator, we need a bias contraction at the top level
    # This handles all components including noise
    overall_bias_contraction = load_bias_contraction(str(folder_path), required=True)

    # Create multipole emulator with a placeholder bias contraction
    # (The actual contraction happens at the MultipoleNoiseEmulator level)
    def placeholder_contraction(biases, stacked_array):
        # This is never called - MultipoleNoiseEmulator uses its own bias_contraction
        raise NotImplementedError("This should not be called")

    multipole_emulator = MultipoleEmulators(P11_emulator, Ploop_emulator, Pct_emulator, placeholder_contraction)

    # Return the MultipoleNoiseEmulator instance with bias contraction
    return MultipoleNoiseEmulator(multipole_emulator, noise_emulator, overall_bias_contraction)
