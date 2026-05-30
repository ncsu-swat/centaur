
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_union1d_inputs():
    list_of_inputs = []

    # Input 1: Basic positive integers, padded
    ar1 = np.array([1, 2, 3], dtype=np.int32)
    ar2 = np.array([3, 4, 5], dtype=np.int32)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "size": 6,
        "fill_value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative values, truncated size
    ar1 = np.array([-3, -2, -1], dtype=np.float32)
    ar2 = np.array([-5, -4, -3], dtype=np.float32)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "size": 3,
        "fill_value": -1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float values, size matches the exact union size
    ar1 = np.array([0.5, 1.5, 2.5], dtype=np.float32)
    ar2 = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "size": 4,
        "fill_value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Large padded size
    ar1 = np.array([10, 20], dtype=np.int32)
    ar2 = np.array([20, 30], dtype=np.int32)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "size": 8,
        "fill_value": -999.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Empty first array
    ar1 = np.array([], dtype=np.int32)
    ar2 = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "size": 5,
        "fill_value": -1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float64 inputs with distinct values
    ar1 = np.array([1.1, 2.2, 3.3], dtype=np.float64)
    ar2 = np.array([4.4, 5.5], dtype=np.float64)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "size": 5,
        "fill_value": 9.9
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Duplicate values within input arrays
    ar1 = np.array([1, 1, 1], dtype=np.int32)
    ar2 = np.array([2, 2, 2], dtype=np.int32)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "size": 4,
        "fill_value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Int64 inputs, negative and positive
    ar1 = np.array([-10, 10], dtype=np.int64)
    ar2 = np.array([-20, 20], dtype=np.int64)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "size": 6,
        "fill_value": -99.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Completely overlapping arrays
    ar1 = np.array([1, 2, 3], dtype=np.float32)
    ar2 = np.array([1, 2, 3], dtype=np.float32)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "size": 5,
        "fill_value": -1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Disjoint arrays, high padding
    ar1 = np.array([5, 6, 7], dtype=np.float32)
    ar2 = np.array([1, 2, 3], dtype=np.float32)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "size": 10,
        "fill_value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.union1d_3"] = jax_numpy_union1d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.union1d_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.union1d_3'.")


check_valid('jax.numpy.union1d', generated_inputs['jax.numpy.union1d_3'], lib="jax", suffix=3)
