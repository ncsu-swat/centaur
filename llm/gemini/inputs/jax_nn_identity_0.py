
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def identity_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with positive, negative, and zero values
    x = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 2: 2D float32 array (matrix)
    x = np.random.randn(3, 4).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 3: 3D float64 array
    x = np.random.randn(2, 2, 2).astype(np.float64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 4: 1D int32 array with negative values
    x = np.array([-10, -5, 0, 5, 10], dtype=np.int32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 5: 4D float16 array
    x = np.random.randn(2, 3, 4, 5).astype(np.float16)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 6: 0D array (scalar)
    x = np.array(42.0, dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 7: Boolean array
    x = np.array([[True, False], [False, True]], dtype=np.bool_)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 8: Complex64 array
    x = np.array([1.0 + 2.0j, -1.0 - 3.0j], dtype=np.complex64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 9: Large multi-dimensional float32 array
    x = np.random.randn(10, 10, 10).astype(np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 10: Array containing special values (inf, -inf, nan)
    x = np.array([np.inf, -np.inf, np.nan, 0.0], dtype=np.float32)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    # Input 11: 1D int64 array with large integers
    x = np.array([2**31, -2**31, 0], dtype=np.int64)
    list_of_inputs.append({"x": copy.deepcopy(x)})

    return list_of_inputs

generated_inputs["jax.nn.identity"] = identity_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.nn.identity' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.nn.identity'.")


check_valid('jax.nn.identity', generated_inputs['jax.nn.identity'], lib="jax", suffix=0)
