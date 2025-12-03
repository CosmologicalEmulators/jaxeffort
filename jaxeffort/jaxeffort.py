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
    w0waCDMCosmology,
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
    def __init__(self, P11: MLP, Ploop: MLP, Pct: MLP, bias_combination: callable,
                 stoch_model: callable, jacobian_bias_combination: callable = None):
        """
        Initializes the MultipoleEmulators class with three MLP instances, bias combination,
        and stochastic model.

        Args:
            P11 (MLP): MLP instance for P11 emulator.
            Ploop (MLP): MLP instance for Ploop emulator.
            Pct (MLP): MLP instance for Pct emulator.
            bias_combination (callable): Bias combination function for the multipole.
                Maps bias parameters to linear combination coefficients.
            stoch_model (callable): Stochastic model function that takes k-grid and returns
                stochastic component arrays.
            jacobian_bias_combination (callable, optional): Analytical Jacobian of bias combination
                                                           with respect to bias parameters.
        """
        self.P11 = P11
        self.Ploop = Ploop
        self.Pct = Pct
        self.bias_combination = bias_combination
        self.stoch_model = stoch_model
        self.jacobian_bias_combination = jacobian_bias_combination

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
        Get P_ℓ using the multipole's bias combination function.
        Matches Effort.jl where BiasCombination is at PℓEmulator level only.

        This method uses JIT compilation for performance.
        Includes stochastic components via StochModel.
        """
        if self.bias_combination is None:
            raise ValueError("bias_combination is required to compute P_ℓ with biases")

        # Create JIT-compiled version on first call
        if not hasattr(self, '_jit_get_Pl'):
            @partial(jax.jit, static_argnums=(0,))
            def _jit_get_Pl(self, cosmology, biases, D):
                P11_comp, Ploop_comp, Pct_comp = self.get_multipole_components(cosmology, D)
                # Add stochastic components - StochModel takes k-grid as input
                stoch_comp = self.stoch_model(self.P11.k_grid)
                stacked_array = jnp.hstack((P11_comp, Ploop_comp, Pct_comp, stoch_comp))
                # BiasCombination returns coefficient vector
                biases_vec = self.bias_combination(biases)
                # Compute P_ℓ = stacked_array @ biases_vec
                return stacked_array @ biases_vec
            self._jit_get_Pl = _jit_get_Pl

        return self._jit_get_Pl(self, cosmology, biases, D)

    def get_Pl_no_bias(self, cosmology, D):
        """Get raw components without bias combination."""
        P11_output, Ploop_output, Pct_output = self.get_multipole_components(cosmology, D)
        return jnp.hstack((P11_output, Ploop_output, Pct_output))

    def get_Pl_jacobian(self, cosmology, biases, D):
        """
        Compute both the power spectrum multipole P_ℓ and its Jacobian with respect to
        bias parameters.

        This function is optimized for inference workflows where both the power spectrum
        and its derivatives are needed (e.g., gradient-based MCMC, Fisher forecasts).
        It computes both quantities in a single pass, avoiding redundant neural network
        evaluations.

        The Jacobian is computed using the analytical derivative of the bias combination
        function, which is significantly faster than automatic differentiation.

        Args:
            cosmology: Array of cosmological parameters.
            biases: Array of bias parameters.
            D: Growth factor value.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: (P_ℓ, ∂P_ℓ/∂b) where:
                - P_ℓ: Power spectrum multipole values
                - ∂P_ℓ/∂b: Jacobian matrix with respect to bias parameters

        Raises:
            ValueError: If jacobian_bias_combination was not provided during initialization.
        """
        if self.jacobian_bias_combination is None:
            raise ValueError(
                "jacobian_bias_combination is required to compute Jacobian. "
                "The emulator was loaded without the jacbiascombination.py file. "
                "Please ensure the file exists in the emulator directory."
            )

        # Create JIT-compiled version on first call
        if not hasattr(self, '_jit_get_Pl_jacobian'):
            @partial(jax.jit, static_argnums=(0,))
            def _jit_get_Pl_jacobian(self, cosmology, biases, D):
                # Get all components (single NN evaluation)
                P11_comp, Ploop_comp, Pct_comp = self.get_multipole_components(cosmology, D)
                # Add stochastic components
                stoch_comp = self.stoch_model(self.P11.k_grid)
                stacked_array = jnp.hstack((P11_comp, Ploop_comp, Pct_comp, stoch_comp))

                # Compute bias combination and its Jacobian
                # BiasCombination returns coefficient vector (no stacked_array arg)
                biases_vec = self.bias_combination(biases)
                jac_biases = self.jacobian_bias_combination(biases)

                # Pl = stacked_array @ biases_vec
                # ∂Pl/∂b = stacked_array @ jac_biases
                Pl = stacked_array @ biases_vec
                Pl_jac = stacked_array @ jac_biases

                return Pl, Pl_jac

            self._jit_get_Pl_jacobian = _jit_get_Pl_jacobian

        return self._jit_get_Pl_jacobian(self, cosmology, biases, D)


def load_preprocessing(root_path, filename):
    """Load postprocessing function from Python file."""
    spec = importlib.util.spec_from_file_location(filename, root_path + "/" + filename + ".py")
    test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test)
    return test.postprocessing


def load_bias_combination(root_path, filename="biascombination", required=True):
    """
    Load bias combination function from Python file.

    This function loads the bias combination (previously called bias contraction)
    which maps bias parameters to the linear combination coefficients for the
    power spectrum components.

    Args:
        root_path (str): Root path where the file is located.
        filename (str): Filename without extension. Default: "biascombination".
        required (bool): Whether to raise an error if file is not found.

    Returns:
        callable: The bias combination function.
    """
    filepath = root_path + "/" + filename + ".py"
    import os
    if not os.path.exists(filepath):
        if required:
            raise FileNotFoundError(f"Bias combination file not found: {filepath}")
        return None

    spec = importlib.util.spec_from_file_location(filename, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Try to get BiasCombination (matching Effort.jl) or BiasContraction (legacy)
    if hasattr(module, 'BiasCombination'):
        return module.BiasCombination
    elif hasattr(module, 'BiasContraction'):
        return module.BiasContraction
    elif hasattr(module, 'biascontraction'):
        return module.biascontraction
    else:
        if required:
            raise AttributeError(f"No BiasCombination or BiasContraction function found in {filepath}")
        return None


def load_jacobian_bias_combination(root_path, filename="jacbiascombination", required=False):
    """
    Load Jacobian bias combination function from Python file.

    This function returns the analytical Jacobian of the bias combination with respect
    to bias parameters, which is significantly faster than automatic differentiation.

    Args:
        root_path (str): Root path where the file is located.
        filename (str): Filename without extension. Default: "jacbiascombination".
        required (bool): Whether to raise an error if file is not found. Default: False.

    Returns:
        callable or None: The Jacobian bias combination function, or None if not found
                         and not required.
    """
    filepath = root_path + "/" + filename + ".py"
    import os
    if not os.path.exists(filepath):
        if required:
            raise FileNotFoundError(f"Jacobian bias combination file not found: {filepath}")
        return None

    spec = importlib.util.spec_from_file_location(filename, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Match Effort.jl naming: JacobianBiasCombination
    if hasattr(module, 'JacobianBiasCombination'):
        return module.JacobianBiasCombination
    else:
        if required:
            raise AttributeError(f"No JacobianBiasCombination function found in {filepath}")
        return None


def load_stoch_model(root_path, filename="stochmodel", required=True):
    """
    Load stochastic model function from Python file.

    The stochastic model computes the stochastic contributions to the power spectrum
    (e.g., shot noise terms). It takes the k-grid as input and returns stochastic
    component arrays.

    Args:
        root_path (str): Root path where the file is located.
        filename (str): Filename without extension. Default: "stochmodel".
        required (bool): Whether to raise an error if file is not found. Default: True.

    Returns:
        callable: The stochastic model function.
    """
    filepath = root_path + "/" + filename + ".py"
    import os
    if not os.path.exists(filepath):
        if required:
            raise FileNotFoundError(f"Stochastic model file not found: {filepath}")
        return None

    spec = importlib.util.spec_from_file_location(filename, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Match Effort.jl naming: StochModel
    if hasattr(module, 'StochModel'):
        return module.StochModel
    elif hasattr(module, 'stoch_model'):
        return module.stoch_model
    else:
        if required:
            raise AttributeError(f"No StochModel function found in {filepath}")
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
    Bias combination is loaded at the multipole level, matching Effort.jl structure.

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

    # Load each component emulator (no bias combination at component level)
    P11_emulator = load_component_emulator(P11_path)
    Ploop_emulator = load_component_emulator(Ploop_path)
    Pct_emulator = load_component_emulator(Pct_path)

    # Load multipole-level bias combination - this is required (matches Effort.jl PℓEmulator)
    multipole_bias_combination = load_bias_combination(str(folder_path), required=True)

    # Load stochastic model - this is required (matches Effort.jl StochModel)
    stoch_model = load_stoch_model(str(folder_path), required=True)

    # Load Jacobian bias combination - this is optional (matches Effort.jl JacobianBiasCombination)
    jacobian_bias_combination = load_jacobian_bias_combination(str(folder_path), required=False)

    # Return the MultipoleEmulators instance with bias combination, stoch model, and optional Jacobian
    return MultipoleEmulators(P11_emulator, Ploop_emulator, Pct_emulator,
                             multipole_bias_combination, stoch_model, jacobian_bias_combination)




