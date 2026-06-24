
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def i0_inputs():
    list_of_inputs = []

    # Input 1: 1D array, positive floats
    x = np.array([0.0, 1.0, 2.5, 5.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D array, negative floats
    x = np.array([-0.5, -1.2, -3.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D array, standard normal distribution (float32)
    x = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D array, uniform distribution (float64)
    x = np.random.uniform(-5.0, 5.0, size=(2, 3, 3)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: Scalar-like (0D array, float64)
    x = np.array(1.5, dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: Large 1D array, linspace range (float64)
    x = np.linspace(-10.0, 10.0, 50).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 2D array of zeros
    x = np.zeros((4, 4), dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 4D array (float32)
    x = np.random.randn(2, 2, 3, 3).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: Small positive values (float32)
    x = np.array([1e-6, 1e-4, 1e-2], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: Mixed positive and negative float64 values
    x = np.array([[-10.0, 0.0, 10.0], [-2.5, 1.1, 3.4]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.scipy.special.i0"] = i0_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.scipy.special.i0' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.scipy.special.i0'.")


check_valid('jax.scipy.special.i0', generated_inputs['jax.scipy.special.i0'], lib="jax", suffix=0)
