
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def unique_all_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D int array
    x = np.array([3, 4, 1, 3, 1], dtype=np.int32)
    input_dict = {
        "x": x,
        "size": 5,
        "fill_value": -1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float array with duplicates
    x = np.array([1.5, -2.3, 1.5, 0.0, -2.3], dtype=np.float32)
    input_dict = {
        "x": x,
        "size": 3,
        "fill_value": -999.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D integer array
    x = np.array([[1, 2], [2, 3]], dtype=np.int32)
    input_dict = {
        "x": x,
        "size": 4,
        "fill_value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float array
    x = np.random.randn(3, 3, 3).astype(np.float32)
    input_dict = {
        "x": x,
        "size": 10,
        "fill_value": -1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative integer array
    x = np.array([-10, -20, -10, -30, -40], dtype=np.int64)
    input_dict = {
        "x": x,
        "size": 6,
        "fill_value": 99.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float64 array
    x = np.arange(10).astype(np.float64)
    input_dict = {
        "x": x,
        "size": 12,
        "fill_value": -0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: All identical values
    x = np.array([5, 5, 5, 5], dtype=np.int32)
    input_dict = {
        "x": x,
        "size": 2,
        "fill_value": -1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D float array with small size
    x = np.ones((2, 2, 2, 2), dtype=np.float32)
    input_dict = {
        "x": x,
        "size": 5,
        "fill_value": -2.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Array with large float values
    x = np.array([1e5, 2e5, 1e5], dtype=np.float64)
    input_dict = {
        "x": x,
        "size": 4,
        "fill_value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Array where unique elements count is exactly equal to size
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2], dtype=np.float32)
    input_dict = {
        "x": x,
        "size": 4,
        "fill_value": -1.23
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.unique_all_3"] = unique_all_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unique_all_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unique_all_3'.")


check_valid('jax.numpy.unique_all', generated_inputs['jax.numpy.unique_all_3'], lib="jax", suffix=3)
