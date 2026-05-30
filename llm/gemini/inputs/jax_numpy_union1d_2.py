
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def union1d_inputs():
    list_of_inputs = []

    # Input 1: Basic integer arrays, size matches exactly
    input_dict = {
        'ar1': np.array([1, 2, 3, 4], dtype=np.int32),
        'ar2': np.array([3, 4, 5, 6], dtype=np.int32),
        'size': 6,
        'fill_value': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Size is smaller than union size (truncation)
    input_dict = {
        'ar1': np.array([10, 20, 30], dtype=np.int32),
        'ar2': np.array([30, 40, 50], dtype=np.int32),
        'size': 3,
        'fill_value': 99
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Size is larger than union size (padding)
    input_dict = {
        'ar1': np.array([1, 2], dtype=np.int32),
        'ar2': np.array([2, 3], dtype=np.int32),
        'size': 5,
        'fill_value': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative values in arrays and negative fill_value
    input_dict = {
        'ar1': np.array([-5, -4, -3], dtype=np.int32),
        'ar2': np.array([-3, -2, -1], dtype=np.int32),
        'size': 5,
        'fill_value': -999
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Duplicate values in input arrays
    input_dict = {
        'ar1': np.array([1, 1, 2, 2], dtype=np.int32),
        'ar2': np.array([2, 2, 3, 3], dtype=np.int32),
        'size': 4,
        'fill_value': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Empty arrays
    input_dict = {
        'ar1': np.array([], dtype=np.int32),
        'ar2': np.array([], dtype=np.int32),
        'size': 3,
        'fill_value': 42
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: One empty array
    input_dict = {
        'ar1': np.array([100, 200], dtype=np.int32),
        'ar2': np.array([], dtype=np.int32),
        'size': 4,
        'fill_value': 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Larger range of numbers
    input_dict = {
        'ar1': np.arange(10, dtype=np.int32),
        'ar2': np.arange(5, 15, dtype=np.int32),
        'size': 15,
        'fill_value': -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Multi-dimensional input (to be flattened)
    input_dict = {
        'ar1': np.array([[1, 2], [3, 4]], dtype=np.int32),
        'ar2': np.array([[3, 4], [5, 6]], dtype=np.int32),
        'size': 8,
        'fill_value': 9
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: High value integers
    input_dict = {
        'ar1': np.array([1000000, 2000000], dtype=np.int32),
        'ar2': np.array([2000000, 3000000], dtype=np.int32),
        'size': 3,
        'fill_value': 9999999
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.union1d_2"] = union1d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.union1d_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.union1d_2'.")


check_valid('jax.numpy.union1d', generated_inputs['jax.numpy.union1d_2'], lib="jax", suffix=2)
