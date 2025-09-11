from typing import Sequence, List

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import json
import importlib.util
from typing import Tuple

jax.config.update("jax_enable_x64", True)

class MLP(nn.Module):
  k_grid:np.array
  features: Sequence[int]
  activations: List[str]
  in_MinMax: np.array
  out_MinMax: np.array
  NN_params: dict
  postprocessing: callable
  emulator_description: dict

  @nn.compact
  def __call__(self, x):
    for i, feat in enumerate(self.features[:-1]):
      if self.activations[i] == "tanh":
        x = nn.tanh(nn.Dense(feat)(x))
      elif self.activations[i] == "relu":
        x = nn.relu(nn.Dense(feat)(x))
      # Add more activation functions as needed
    x = nn.Dense(self.features[-1])(x)
    return x

  def maximin(self, input):
    return (input - self.in_MinMax[:,0]) / (self.in_MinMax[:,1] - self.in_MinMax[:,0])

  def inv_maximin(self, output):
    return output * (self.out_MinMax[:,1] - self.out_MinMax[:,0]) + self.out_MinMax[:,0]

  def get_component(self, input, D):
    """Get raw component output without bias contraction, matching Effort.jl's get_component."""
    norm_input = self.maximin(input)
    norm_model_output = self.apply(self.NN_params, norm_input)
    model_output = self.inv_maximin(norm_model_output)
    processed_model_output = self.postprocessing(input, model_output, D, self)
    reshaped_output = processed_model_output.reshape(
        (len(self.k_grid), int(len(processed_model_output) / len(self.k_grid))), order = "F"
    )
    return reshaped_output


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

    def get_multipole_components(self, inputs: np.array, D) -> Tuple[np.array, np.array, np.array]:
        """
        Computes the raw component outputs for all three emulators given an input array.

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
        """
        P11_comp, Ploop_comp, Pct_comp = self.get_multipole_components(cosmology, D)
        stacked_array = jnp.hstack((P11_comp, Ploop_comp, Pct_comp))
        
        # Use multipole-level bias contraction (matches Effort.jl's PℓEmulator.BiasContraction)
        return self.bias_contraction(biases, stacked_array)

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
        """Get P_ℓ with noise, using bias contraction."""
        # Get all components
        P11_comp, Ploop_comp, Pct_comp = self.multipole_emulator.get_multipole_components(cosmology, D)
        Noise_comp = self.noise_emulator.get_component(cosmology, D)
        stacked_array = jnp.hstack((P11_comp, Ploop_comp, Pct_comp, Noise_comp))
        
        # Use the overall bias contraction
        return self.bias_contraction(biases, stacked_array)

    def get_Pl_no_bias(self, cosmology, D):
        """Get raw components without bias contraction."""
        P11_output, Ploop_output, Pct_output = self.multipole_emulator.get_multipole_components(cosmology, D)
        Noise_output = self.noise_emulator.get_component(cosmology, D)
        return jnp.hstack((P11_output, Ploop_output, Pct_output, Noise_output))


def get_flax_params(nn_dict, weights):
    in_array, out_array = get_in_out_arrays(nn_dict)
    i_array = get_i_array(in_array, out_array)
    params = [get_weight_bias(i_array[j], in_array[j], out_array[j], weights, nn_dict) for j in range(nn_dict["n_hidden_layers"]+1)]
    layer = ["layer_" + str(j) for j in range(nn_dict["n_hidden_layers"]+1)]
    return dict(zip(layer, params))

def get_weight_bias(i, n_in, n_out, weight_bias, nn_dict):
    weight = np.reshape(weight_bias[i:i+n_out*n_in], (n_in, n_out))
    bias = weight_bias[i+n_out*n_in:i+n_out*n_in+n_out]
    i += n_out*n_in+n_out
    return {'kernel': weight, 'bias': bias}, i

def get_in_out_arrays(nn_dict):
    n = nn_dict["n_hidden_layers"]
    in_array = np.zeros(n+1, dtype=int)
    out_array = np.zeros(n+1, dtype=int)
    in_array[0] = nn_dict["n_input_features"]
    out_array[-1] = nn_dict["n_output_features"]
    for i in range(n):
        in_array[i+1] = nn_dict["layers"]["layer_" + str(i+1)]["n_neurons"]
        out_array[i] = nn_dict["layers"]["layer_" + str(i+1)]["n_neurons"]
    return in_array, out_array

def get_i_array(in_array, out_array):
    i_array = np.empty_like(in_array)
    i_array[0] = 0
    for i in range(1, len(i_array)):
        i_array[i] = i_array[i-1] + in_array[i-1]*out_array[i-1] + out_array[i-1]
    return i_array

def load_weights(nn_dict, weights):
    in_array, out_array = get_in_out_arrays(nn_dict)
    i_array = get_i_array(in_array, out_array)
    variables = {'params': {}}
    i = 0
    for j in range(nn_dict["n_hidden_layers"]+1):
        layer_params, i = get_weight_bias(i_array[j], in_array[j], out_array[j], weights, nn_dict)
        variables['params']["Dense_" + str(j)] = layer_params
    return variables

def load_activation_function(nn_dict):
    list_activ_func = []
    for j in range(nn_dict["n_hidden_layers"]):
        list_activ_func.append(nn_dict["layers"]["layer_" + str(j+1)]["activation_function"])
    return list_activ_func

def load_number_neurons(nn_dict):
    list_n_neurons = []
    for j in range(nn_dict["n_hidden_layers"]):
        list_n_neurons.append(nn_dict["layers"]["layer_" + str(j+1)]["n_neurons"])
    list_n_neurons.append(nn_dict["n_output_features"])
    return list_n_neurons

def load_preprocessing(root_path, filename):
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
    """Load a component emulator (P11, Ploop, Pct, or Noise) without bias contraction."""
    in_MinMax = jnp.load(folder_path + "inminmax.npy")

    f = open(folder_path + '/nn_setup.json')

    # returns JSON object as
    # a dictionary
    NN_dict = json.load(f)
    f.close()

    #spec = importlib.util.spec_from_file_location("postprocessing", "postprocessing.py")
    #test = importlib.util.module_from_spec(spec)
    #spec.loader.exec_module(test)

    postprocessing = load_preprocessing(folder_path, "postprocessing")

    activation_function_list = load_activation_function(NN_dict)
    list_n_neurons = load_number_neurons(NN_dict)

    k_grid = jnp.load(folder_path + "k.npy")
    weights = jnp.load(folder_path + "weights.npy")
    out_MinMax = jnp.load(folder_path + "outminmax.npy")
    variables = load_weights(NN_dict, weights)

    # Components don't have bias contraction - that's at PℓEmulator level
    return MLP(k_grid, list_n_neurons, activation_function_list, in_MinMax, out_MinMax, variables, postprocessing, NN_dict["emulator_description"])

def load_multipole_emulator(folder_path: str) -> MultipoleEmulators:
    """
    Loads the three multipole emulators (P11, Ploop, Pct) from their respective subfolders.
    Bias contraction is loaded at the multipole level, matching Effort.jl structure.

    Args:
        folder_path (str): The path to the folder containing the subfolders `11`, `loop`, and `ct`.

    Returns:
        MultipoleEmulators: An instance of the MultipoleEmulators class containing the loaded emulators.
    """
    # Define subfolder paths
    P11_path = f"{folder_path}/11/"
    Ploop_path = f"{folder_path}/loop/"
    Pct_path = f"{folder_path}/ct/"

    # Load each component emulator (no bias contraction at component level)
    P11_emulator = load_component_emulator(P11_path)
    Ploop_emulator = load_component_emulator(Ploop_path)
    Pct_emulator = load_component_emulator(Pct_path)
    
    # Load multipole-level bias contraction - this is required (matches Effort.jl PℓEmulator)
    multipole_bias_contraction = load_bias_contraction(folder_path, required=True)

    # Return the MultipoleEmulators instance with bias contraction
    return MultipoleEmulators(P11_emulator, Ploop_emulator, Pct_emulator, multipole_bias_contraction)

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
    # Define subfolder paths
    P11_path = f"{folder_path}/11/"
    Ploop_path = f"{folder_path}/loop/"
    Pct_path = f"{folder_path}/ct/"
    noise_path = f"{folder_path}/st/"
    
    # Load component emulators (no bias contraction at component level)
    P11_emulator = load_component_emulator(P11_path)
    Ploop_emulator = load_component_emulator(Ploop_path)
    Pct_emulator = load_component_emulator(Pct_path)
    noise_emulator = load_component_emulator(noise_path)
    
    # For MultipoleNoiseEmulator, we need a bias contraction at the top level
    # This handles all components including noise
    overall_bias_contraction = load_bias_contraction(folder_path, required=True)
    
    # Create multipole emulator with a placeholder bias contraction
    # (The actual contraction happens at the MultipoleNoiseEmulator level)
    def placeholder_contraction(biases, stacked_array):
        # This is never called - MultipoleNoiseEmulator uses its own bias_contraction
        raise NotImplementedError("This should not be called")
    
    multipole_emulator = MultipoleEmulators(P11_emulator, Ploop_emulator, Pct_emulator, placeholder_contraction)
    
    # Return the MultipoleNoiseEmulator instance with bias contraction
    return MultipoleNoiseEmulator(multipole_emulator, noise_emulator, overall_bias_contraction)
