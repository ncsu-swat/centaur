
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_atleast_1d_inputs():
    list_of_inputs = []

    # Input 1: 0-D array (scalar) - float
    arys = np.array(1.5, dtype=np.float32)
    list_of_inputs.append({"arys": copy.deepcopy(arys)})

    # Input 2: 0-D array (scalar) - negative integer
    arys = np.array(-10, dtype=np.int32)
    list_of_inputs.append({"arys": copy.deepcopy(arys)})

    # Input 3: 1-D array - float
    arys = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    list_of_inputs.append({"arys": copy.deepcopy(arys)})

    # Input 4: 1-D array - integer with negative values
    arys = np.array([-1, 0, 1], dtype=np.int32)
    list_of_inputs.append({"arys": copy.deepcopy(arys)})

    # Input 5: 2-D array
    arys = np.random.randn(2, 3).astype(np.float32)
    list_of_inputs.append({"arys": copy.deepcopy(arys)})

    # Input 6: 3-D array
    arys = np.random.randn(2, 2, 2).astype(np.float32)
    list_of_inputs.append({"arys": copy.deepcopy(arys)})

    # Input 7: Empty 1-D array
    arys = np.array([], dtype=np.float32)
    list_of_inputs.append({"arys": copy.deepcopy(arys)})

    # Input 8: Boolean array
    arys = np.array([True, False, True], dtype=bool)
    list_of_inputs.append({"arys": copy.deepcopy(arys)})

    # Input 9: High-dimensional array (4-D)
    arys = np.random.randn(1, 2, 1, 3).astype(np.float32)
    list_of_inputs.append({"arys": copy.deepcopy(arys)})

    # Input 10: Complex float array
    arys = np.array([1.0 + 2.0j, -3.0 - 4.0j], dtype=np.complex64)
    list_of_inputs.append({"arys": copy.deepcopy(arys)})

    # Input 11: uint8 array
    arys = np.array([0, 127, 255], dtype=np.uint8)
    list_of_inputs.append({"arys": copy.deepcopy(arys)})

    return list_of_inputs

generated_inputs["jax.numpy.atleast_1d_1"] = jax_numpy_atleast_1d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.atleast_1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.atleast_1d_1'.")


check_valid('jax.numpy.atleast_1d', generated_inputs['jax.numpy.atleast_1d_1'], lib="jax", suffix=1)
