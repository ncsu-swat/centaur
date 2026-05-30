
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def log_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive values
    x = np.array([1.0, 2.0, 3.0, 10.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D float64 array with positive values
    x = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 1D int32 array with positive values (will be promoted to inexact)
    x = np.array([1, 4, 9, 16], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D float32 array
    x = np.random.uniform(0.1, 10.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 0D array (scalar array)
    x = np.array(5.0, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 2D float16 array with positive values
    x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: 1D complex64 array (log of complex numbers)
    x = np.array([1.0 + 1.0j, 2.0 - 1.0j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 4D float32 array
    x = np.random.uniform(1.0, 100.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 1D float32 array with very small positive values
    x = np.array([1e-5, 1e-10, 1e-20], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 1D float32 array with very large positive values
    x = np.array([1e5, 1e10, 1e20], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.log_1"] = log_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.log_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.log_1'.")


check_valid('jax.numpy.log', generated_inputs['jax.numpy.log_1'], lib="jax", suffix=1)
