
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def array_inputs():
    list_of_inputs = []

    # Input 1: 1D list of ints, int32 dtype
    input_dict = {
        "object": [1, 2, 3, 4],
        "dtype": np.dtype(np.int32),
        "copy": True,
        "order": "K",
        "ndmin": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D list of floats with negative values, float32 dtype
    input_dict = {
        "object": [[-1.5, 2.0], [3.2, -4.8]],
        "dtype": np.dtype(np.float32),
        "copy": False,
        "order": "K",
        "ndmin": 2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Empty list, float64 dtype, higher ndmin
    input_dict = {
        "object": [],
        "dtype": np.dtype(np.float64),
        "copy": True,
        "order": "K",
        "ndmin": 3,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D list of booleans, bool dtype
    input_dict = {
        "object": [True, False, True, True],
        "dtype": np.dtype(np.bool_),
        "copy": False,
        "order": "K",
        "ndmin": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D nested list of ints, int64 dtype
    input_dict = {
        "object": [[[1, -2], [3, 4]], [[-5, 6], [7, -8]]],
        "dtype": np.dtype(np.int64),
        "copy": True,
        "order": "K",
        "ndmin": 3,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D list with ndmin expanding dimensions to 2D
    input_dict = {
        "object": [10, 20, 30],
        "dtype": np.dtype(np.uint32),
        "copy": False,
        "order": "K",
        "ndmin": 2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D list of complex numbers, complex64 dtype
    input_dict = {
        "object": [[1.0 + 2.0j, -3.0j], [4.5 - 1.2j, 0.0 + 0.0j]],
        "dtype": np.dtype(np.complex64),
        "copy": True,
        "order": "K",
        "ndmin": 2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D list of integers with int16 dtype
    input_dict = {
        "object": [-32768, 0, 32767],
        "dtype": np.dtype(np.int16),
        "copy": True,
        "order": "K",
        "ndmin": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D nested list of floats, float16 dtype
    input_dict = {
        "object": [[[[1.0], [2.0]], [[3.0], [4.0]]]],
        "dtype": np.dtype(np.float16),
        "copy": False,
        "order": "K",
        "ndmin": 4,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D list of unsigned 8-bit integers
    input_dict = {
        "object": [0, 128, 255],
        "dtype": np.dtype(np.uint8),
        "copy": True,
        "order": "K",
        "ndmin": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.array_2"] = array_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.array_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.array_2'.")


check_valid('jax.numpy.array', generated_inputs['jax.numpy.array_2'], lib="jax", suffix=2)
