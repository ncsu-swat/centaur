
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def ediff1d_inputs():
    list_of_inputs = []

    # Input 1: 1D float array with positive floats
    input_dict = {
        "ary": np.array([1.0, 2.5, 4.0, 7.5], dtype=np.float32),
        "to_end": 10.0,
        "to_begin": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D integer array with negative float bounds
    input_dict = {
        "ary": np.array([10, 20, 15, 30, 25], dtype=np.int32),
        "to_end": -5.5,
        "to_begin": -1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, flattened during ediff1d
    input_dict = {
        "ary": np.array([[-2.0, 3.5], [4.0, -1.5]], dtype=np.float64),
        "to_end": 100.25,
        "to_begin": -50.75,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array with zeros
    input_dict = {
        "ary": np.zeros((2, 3, 4), dtype=np.float32),
        "to_end": 0.0,
        "to_begin": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D array of single element
    input_dict = {
        "ary": np.array([42.0], dtype=np.float32),
        "to_end": 5.0,
        "to_begin": 1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty 1D array
    input_dict = {
        "ary": np.array([], dtype=np.float64),
        "to_end": 9.99,
        "to_begin": -9.99,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float64 1D array with negative numbers
    input_dict = {
        "ary": np.array([-10.5, -20.25, -5.0, 0.0], dtype=np.float64),
        "to_end": 12.34,
        "to_begin": -43.21,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: High-dimensional array (4D)
    input_dict = {
        "ary": np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2),
        "to_end": 256.0,
        "to_begin": -128.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large values
    input_dict = {
        "ary": np.array([1e6, 2e6, 4e6], dtype=np.float64),
        "to_end": 1e7,
        "to_begin": 1e5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Array with inf and nan
    input_dict = {
        "ary": np.array([np.nan, 1.0, np.inf, 2.0], dtype=np.float32),
        "to_end": float("inf"),
        "to_begin": float("-inf"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.ediff1d_3"] = ediff1d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.ediff1d_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.ediff1d_3'.")


check_valid('jax.numpy.ediff1d', generated_inputs['jax.numpy.ediff1d_3'], lib="jax", suffix=3)
