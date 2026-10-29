
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def ones_like_inputs():
    list_of_inputs = []

    # Input 1: 1D array, int32, shape overridden to 1D list
    input_dict = {
        "a": np.array([1, 2, 3], dtype=np.int32),
        "dtype": np.dtype(np.int32),
        "shape": [5],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, float32, shape overridden to 2D list
    input_dict = {
        "a": np.random.randn(2, 3).astype(np.float32),
        "dtype": np.dtype(np.float32),
        "shape": [4, 2],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array with negative values, float64, shape overridden to 1D list
    input_dict = {
        "a": np.array(
            [[[-1.5, -2.5], [3.0, 4.0]], [[-5.1, 6.2], [-7.3, 8.4]]],
            dtype=np.float64,
        ),
        "dtype": np.dtype(np.float64),
        "shape": [8],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Scalar-like 0D array, bool dtype, shape overridden to 3D list
    input_dict = {
        "a": np.array(True, dtype=np.bool_),
        "dtype": np.dtype(np.bool_),
        "shape": [2, 2, 3],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Complex numbers, complex64 dtype, shape overridden to 2D list
    input_dict = {
        "a": np.array([1.0 + 2.0j, -3.0 - 4.0j], dtype=np.complex64),
        "dtype": np.dtype(np.complex64),
        "shape": [3, 3],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array, int16 dtype, shape overridden to empty list (scalar)
    input_dict = {
        "a": np.zeros((2, 1, 3, 4), dtype=np.int16),
        "dtype": np.dtype(np.int16),
        "shape": [],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array, uint8 dtype, shape overridden to 2D list
    input_dict = {
        "a": np.array([[0, 255], [128, 64]], dtype=np.uint8),
        "dtype": np.dtype(np.uint8),
        "shape": [1, 5],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array with negative ints, int64 dtype, shape overridden to 4D list
    input_dict = {
        "a": np.array([-10, -20, 0, 30], dtype=np.int64),
        "dtype": np.dtype(np.int64),
        "shape": [1, 2, 1, 4],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float array, changed to int32 dtype, shape overridden to 2D list
    input_dict = {
        "a": np.ones((3, 3), dtype=np.float32),
        "dtype": np.dtype(np.int32),
        "shape": [2, 4],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array, complex128 dtype, shape overridden to 3D list
    input_dict = {
        "a": np.array([0.0 - 1.0j, 2.5 + 0.0j], dtype=np.complex128),
        "dtype": np.dtype(np.complex128),
        "shape": [2, 1, 2],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D int8 array, uint32 dtype, shape overridden to 1D list
    input_dict = {
        "a": np.array([[[-128, 127]]], dtype=np.int8),
        "dtype": np.dtype(np.uint32),
        "shape": [6],
    }
    list_of_inputs.append(copy.deepcopy(input_dict))


    return list_of_inputs


generated_inputs["jax.numpy.ones_like_3"] = ones_like_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ones_like_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ones_like_3'.")


check_valid('jax.numpy.ones_like', generated_inputs['jax.numpy.ones_like_3'], lib="jax", suffix=3)
