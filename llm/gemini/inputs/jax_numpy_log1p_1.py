
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def log1p_inputs():
    list_of_inputs = []

    # Input 1: 1D array with positive float32 values
    x = np.array([2.0, 5.0, 9.0, 4.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 1D array with very small float32 values close to 0
    x = np.array([1e-4, 1e-6, 2e-10], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 2D array with float64 values
    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 3D array with mixed range values > -1
    x = np.random.uniform(-0.5, 5.0, size=(2, 3, 4)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 1D array with int32 values
    x = np.array([0, 1, 2, 10, 100], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 0D array (scalar)
    x = np.array(0.5, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: Large 2D array of float32 values
    x = np.random.uniform(0.1, 10.0, size=(100, 100)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: 4D array of float32 values
    x = np.random.uniform(-0.99, 1.0, size=(2, 2, 2, 2)).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: 1D array with small negative float64 values close to 0
    x = np.array([-1e-5, -1e-12, -2e-16], dtype=np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: 1D complex64 array
    x = np.array([1.0 + 1.0j, 2.0 - 3.0j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.numpy.log1p_1"] = log1p_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.log1p_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.log1p_1'.")


check_valid('jax.numpy.log1p', generated_inputs['jax.numpy.log1p_1'], lib="jax", suffix=1)
