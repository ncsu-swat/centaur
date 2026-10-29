
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def union1d_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D integer arrays, size larger than union
    ar1 = np.array([1, 2, 3, 4], dtype=np.int32)
    ar2 = np.array([3, 4, 5, 6], dtype=np.int32)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "size": 8,
        "fill_value": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Truncated size (size smaller than union)
    ar1 = np.array([10, 20, 30], dtype=np.int64)
    ar2 = np.array([40, 50, 60], dtype=np.int64)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "size": 4,
        "fill_value": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative numbers and negative fill_value
    ar1 = np.array([-5, -3, -1, 0], dtype=np.int32)
    ar2 = np.array([-2, -1, 2], dtype=np.int32)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "size": 7,
        "fill_value": -99,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float arrays with integer fill_value
    ar1 = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    ar2 = np.array([2.5, 4.5], dtype=np.float32)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "size": 6,
        "fill_value": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Multi-dimensional arrays (2D)
    ar1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    ar2 = np.array([[3, 4], [5, 6]], dtype=np.int32)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "size": 6,
        "fill_value": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Disjoint arrays
    ar1 = np.array([1, 3, 5, 7], dtype=np.int32)
    ar2 = np.array([2, 4, 6, 8], dtype=np.int32)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "size": 10,
        "fill_value": 999,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Identical arrays with duplicates
    ar1 = np.array([2, 2, 2, 2], dtype=np.int32)
    ar2 = np.array([2, 2], dtype=np.int32)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "size": 3,
        "fill_value": -1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Empty array inputs
    ar1 = np.array([], dtype=np.int32)
    ar2 = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "size": 5,
        "fill_value": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Size zero
    ar1 = np.array([1, 2, 3], dtype=np.int32)
    ar2 = np.array([4, 5, 6], dtype=np.int32)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "size": 0,
        "fill_value": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D and 1D inputs combined
    ar1 = np.arange(8, dtype=np.int32).reshape(2, 2, 2)
    ar2 = np.array([7, 8, 9, 10], dtype=np.int32)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "size": 12,
        "fill_value": -5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Exact union size
    ar1 = np.array([100, 200], dtype=np.int64)
    ar2 = np.array([200, 300], dtype=np.int64)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "size": 3,
        "fill_value": 0,
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
