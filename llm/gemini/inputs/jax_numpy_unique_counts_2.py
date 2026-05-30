
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def unique_counts_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array, size matches exact number of unique elements
    x = np.array([3, 4, 1, 3, 1], dtype=np.int32)
    input_dict = {
        "x": x,
        "size": 3,
        "fill_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: size is larger than unique elements, padding with positive integer
    x = np.array([1, 2, 2, 1], dtype=np.int32)
    input_dict = {
        "x": x,
        "size": 5,
        "fill_value": 99
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, size smaller than unique elements
    x = np.array([[10, 20], [30, 10]], dtype=np.int32)
    input_dict = {
        "x": x,
        "size": 2,
        "fill_value": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative values in array, padding with a negative integer
    x = np.array([-5, -10, -5, 0, 5], dtype=np.int32)
    input_dict = {
        "x": x,
        "size": 6,
        "fill_value": -999
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, size matches exact unique elements
    x = np.array([[[1, 2], [1, 2]], [[3, 4], [3, 4]]], dtype=np.int32)
    input_dict = {
        "x": x,
        "size": 4,
        "fill_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large integer values (int64)
    x = np.array([100000, 200000, 100000, 300000], dtype=np.int64)
    input_dict = {
        "x": x,
        "size": 2,
        "fill_value": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Single element array, padding requested
    x = np.array([42], dtype=np.int32)
    input_dict = {
        "x": x,
        "size": 3,
        "fill_value": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: All elements are identical
    x = np.array([5, 5, 5, 5, 5], dtype=np.int32)
    input_dict = {
        "x": x,
        "size": 4,
        "fill_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D array with sequential values
    x = np.arange(16, dtype=np.int32).reshape(2, 2, 2, 2)
    input_dict = {
        "x": x,
        "size": 10,
        "fill_value": -5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger array with high repetition
    x = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5] * 10, dtype=np.int32)
    input_dict = {
        "x": x,
        "size": 8,
        "fill_value": 999
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.unique_counts_2"] = unique_counts_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unique_counts_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unique_counts_2'.")


check_valid('jax.numpy.unique_counts', generated_inputs['jax.numpy.unique_counts_2'], lib="jax", suffix=2)
