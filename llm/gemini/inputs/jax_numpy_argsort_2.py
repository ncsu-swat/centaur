
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def argsort_inputs():
    list_of_inputs = []

    # Input 1: 1D array, default-like parameters
    input_dict = {
        "a": np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.int32),
        "axis": -1,
        "stable": True,
        "descending": False,
        "dtype": np.int32,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array with negative numbers, descending
    input_dict = {
        "a": np.array([-10, 20, -30, 40, -50], dtype=np.int64),
        "axis": 0,
        "stable": False,
        "descending": True,
        "dtype": np.int64,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, sort along axis 0
    input_dict = {
        "a": np.array([[5, 2, 9], [1, 7, 3]], dtype=np.int32),
        "axis": 0,
        "stable": True,
        "descending": False,
        "dtype": np.int32,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, sort along axis 1, descending
    input_dict = {
        "a": np.array(
            [[-1, -5, 3], [12, -4, 0], [7, 8, -9]], dtype=np.int32
        ),
        "axis": 1,
        "stable": False,
        "descending": True,
        "dtype": np.int64,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D array, negative axis index
    input_dict = {
        "a": np.array([[10, 4], [3, 8], [-2, 5]], dtype=np.int16),
        "axis": -2,
        "stable": True,
        "descending": False,
        "dtype": np.int32,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, axis 0, int8 dtype
    input_dict = {
        "a": np.array(
            [[[1, -2], [3, 4]], [[-5, 6], [7, -8]]], dtype=np.int8
        ),
        "axis": 0,
        "stable": True,
        "descending": True,
        "dtype": np.int32,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D array, axis 1
    input_dict = {
        "a": np.arange(24, dtype=np.int32).reshape(2, 3, 4)[::-1],
        "axis": 1,
        "stable": False,
        "descending": False,
        "dtype": np.int64,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array, axis 2 (last axis), descending
    input_dict = {
        "a": np.array(
            [[[9, 2, 7], [4, 6, 1]], [[3, 8, 5], [0, -1, 2]]], dtype=np.int64
        ),
        "axis": 2,
        "stable": True,
        "descending": True,
        "dtype": np.int32,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array with duplicates to test stability
    input_dict = {
        "a": np.array([2, 1, 2, 1, 2, 1], dtype=np.int32),
        "axis": -1,
        "stable": True,
        "descending": False,
        "dtype": np.int32,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D array, sort along axis 2
    input_dict = {
        "a": np.array(
            [[[[4, 1], [3, 2]], [[-1, 0], [5, -3]]]], dtype=np.int32
        ),
        "axis": 2,
        "stable": False,
        "descending": False,
        "dtype": np.int64,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 4D array, sort along axis -1, descending
    input_dict = {
        "a": np.arange(16, dtype=np.int64).reshape(2, 2, 2, 2)[::-1],
        "axis": -1,
        "stable": True,
        "descending": True,
        "dtype": np.int32,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.argsort_2"] = argsort_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.argsort_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.argsort_2'.")


check_valid('jax.numpy.argsort', generated_inputs['jax.numpy.argsort_2'], lib="jax", suffix=2)
