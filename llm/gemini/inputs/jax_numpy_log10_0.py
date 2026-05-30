
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def log10_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with standard powers of 10
    x = np.array([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D float32 array with random positive values
    x = np.random.uniform(0.1, 100.0, size=(3, 4)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D float64 array
    x = np.random.uniform(1.0, 50.0, size=(2, 3, 3)).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 1D int32 array (positive integers)
    x = np.array([1, 10, 100, 1000, 10000], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 0D array (scalar)
    x = np.array(15.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 1D float16 array
    x = np.array([0.5, 2.5, 12.5, 62.5], dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: Very small positive numbers (float32)
    x = np.array([1e-15, 1e-10, 1e-5], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: Very large positive numbers (float64)
    x = np.array([1e15, 1e30, 1e50], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 4D float32 array
    x = np.random.uniform(0.5, 1.5, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 2D float32 array with large dimensions
    x = np.random.uniform(10.0, 1000.0, size=(10, 10)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.log10"] = log10_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.log10' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.log10'.")


check_valid('jax.numpy.log10', generated_inputs['jax.numpy.log10'], lib="jax", suffix=0)
