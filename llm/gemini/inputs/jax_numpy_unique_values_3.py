
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def unique_values_inputs():
    list_of_inputs = []

    # Input 1: 1D array, size smaller than unique values
    input_dict = {
        "x": np.array([3.0, 1.0, 2.0, 1.0, 3.0], dtype=np.float32),
        "size": 2,
        "fill_value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array, size larger than unique values (padding needed)
    input_dict = {
        "x": np.array([5.0, 2.0, 8.0, 2.0], dtype=np.float32),
        "size": 5,
        "fill_value": -1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array with negative values
    input_dict = {
        "x": np.array([[-1.0, -2.0], [-2.0, -3.0]], dtype=np.float32),
        "size": 3,
        "fill_value": -9.9
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, float64
    input_dict = {
        "x": np.array([[[1.5, 2.5], [1.5, 3.5]], [[2.5, 3.5], [4.5, 5.5]]], dtype=np.float64),
        "size": 6,
        "fill_value": -0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large 1D array generated randomly, size equal to number of unique values
    x_rand = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32)
    input_dict = {
        "x": x_rand,
        "size": 10,
        "fill_value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Array containing NaN values (unique_values handles NaN with equal_nan=True)
    input_dict = {
        "x": np.array([np.nan, 2.0, np.nan, 1.0], dtype=np.float32),
        "size": 4,
        "fill_value": -99.9
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: High-dimensional array (4D)
    input_dict = {
        "x": np.ones((2, 2, 2, 2), dtype=np.float32),
        "size": 3,
        "fill_value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array with positive and negative infinity
    input_dict = {
        "x": np.array([np.inf, -np.inf, np.inf, 1.0], dtype=np.float32),
        "size": 5,
        "fill_value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single element array
    input_dict = {
        "x": np.array([42.0], dtype=np.float32),
        "size": 2,
        "fill_value": 1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Array with duplicate values and large float size limit
    input_dict = {
        "x": np.array([100.0, 100.0, 200.0, 300.0, 200.0], dtype=np.float32),
        "size": 10,
        "fill_value": -999.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.unique_values_3"] = unique_values_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unique_values_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unique_values_3'.")


check_valid('jax.numpy.unique_values', generated_inputs['jax.numpy.unique_values_3'], lib="jax", suffix=3)
