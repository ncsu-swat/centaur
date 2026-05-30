
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def unique_counts_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D int array with positive numbers, size larger than actual unique count
    input_dict = {
        "x": np.array([1, 2, 2, 3, 3, 3], dtype=np.int32),
        "size": 5,
        "fill_value": np.array(-1, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float array with negative values
    input_dict = {
        "x": np.array([-1.5, -1.5, 2.0, 0.0, 2.0], dtype=np.float32),
        "size": 3,
        "fill_value": np.array(99.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D integer array, size larger than unique elements
    input_dict = {
        "x": np.array([[1, 2], [2, 3]], dtype=np.int64),
        "size": 5,
        "fill_value": np.array(-999, dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Boolean array
    input_dict = {
        "x": np.array([True, False, True, True], dtype=bool),
        "size": 2,
        "fill_value": np.array(False, dtype=bool)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float64 array
    input_dict = {
        "x": np.array([[[1.1, 2.2], [1.1, 3.3]], [[2.2, 4.4], [3.3, 5.5]]], dtype=np.float64),
        "size": 6,
        "fill_value": np.array(0.0, dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large 1D array, size equal to unique count
    input_dict = {
        "x": np.array([10, 20, 10, 30, 40, 50, 20, 30, 40, 50], dtype=np.int32),
        "size": 5,
        "fill_value": np.array(0, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty 1D array
    input_dict = {
        "x": np.array([], dtype=np.int32),
        "size": 3,
        "fill_value": np.array(-1, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D int32 array with negative values
    input_dict = {
        "x": np.array([-100, 100, -100, 50, 0, 50], dtype=np.int32),
        "size": 5,
        "fill_value": np.array(0, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Int64 array instead of uint32
    input_dict = {
        "x": np.array([1, 2, 3, 1, 2, 3], dtype=np.int64),
        "size": 2,
        "fill_value": np.array(0, dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Array with duplicate values, size smaller than actual unique count
    input_dict = {
        "x": np.array([5, 5, 6, 6, 7, 7, 8, 8], dtype=np.int32),
        "size": 2,
        "fill_value": np.array(-1, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.unique_counts_1"] = unique_counts_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unique_counts_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unique_counts_1'.")


check_valid('jax.numpy.unique_counts', generated_inputs['jax.numpy.unique_counts_1'], lib="jax", suffix=1)
