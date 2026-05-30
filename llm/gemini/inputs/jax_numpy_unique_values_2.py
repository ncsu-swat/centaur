
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def unique_values_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D integer array, small size padding needed
    input_dict = {
        "x": np.array([3, 1, 2, 1, 3, 2], dtype=np.int32),
        "size": 5,
        "fill_value": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array with negative values, no padding needed
    input_dict = {
        "x": np.array([-10, -5, -10, 0, 5, 5], dtype=np.int32),
        "size": 3,
        "fill_value": 99
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D integer array, flattened by unique_values
    input_dict = {
        "x": np.array([[1, 2, 3], [3, 2, 1]], dtype=np.int32),
        "size": 4,
        "fill_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large range 1D array, int64
    input_dict = {
        "x": np.array([100, 200, 100, 300, 400], dtype=np.int64),
        "size": 6,
        "fill_value": -9999
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array
    input_dict = {
        "x": np.arange(8).reshape(2, 2, 2).astype(np.int32),
        "size": 10,
        "fill_value": 999
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float array (treated as tensor) with integer size and fill_value
    input_dict = {
        "x": np.array([1.5, 2.5, 1.5, 3.5], dtype=np.float32),
        "size": 5,
        "fill_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array with all identical elements
    input_dict = {
        "x": np.array([42, 42, 42, 42], dtype=np.int32),
        "size": 3,
        "fill_value": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: High-dimensional 4D array
    input_dict = {
        "x": np.random.randint(0, 5, size=(2, 2, 2, 2), dtype=np.int32),
        "size": 8,
        "fill_value": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Array with binary-like values
    input_dict = {
        "x": np.array([1, 0, 1, 1, 0, 0, 1], dtype=np.int32),
        "size": 4,
        "fill_value": 9
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Array containing a single element
    input_dict = {
        "x": np.array([7], dtype=np.int32),
        "size": 2,
        "fill_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.unique_values_2"] = unique_values_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unique_values_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unique_values_2'.")


check_valid('jax.numpy.unique_values', generated_inputs['jax.numpy.unique_values_2'], lib="jax", suffix=2)
