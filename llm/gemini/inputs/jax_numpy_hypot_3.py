
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def hypot_inputs():
    list_of_inputs = []

    # Input 1: Scalar python integers
    input_dict = {
        "x1": 3,
        "x2": 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D numpy arrays of int32
    input_dict = {
        "x1": np.array([3, 5, 8], dtype=np.int32),
        "x2": np.array([4, 12, 15], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D numpy arrays of int32, same shape
    input_dict = {
        "x1": np.array([[3, 4], [5, 12]], dtype=np.int32),
        "x2": np.array([[4, 3], [12, 5]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative integer arrays, int32
    input_dict = {
        "x1": np.array([-3, -5, -8], dtype=np.int32),
        "x2": np.array([-4, -12, -15], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Scalar np.int64 and 1D np.int64 array
    input_dict = {
        "x1": np.int64(10),
        "x2": np.array([20, 30, 40], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting 2D arrays (1, 3) and (3, 1), int32
    input_dict = {
        "x1": np.array([[1, 2, 3]], dtype=np.int32),
        "x2": np.array([[4], [5], [6]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D numpy arrays, int16
    input_dict = {
        "x1": np.array([[[1, 2], [3, 4]]], dtype=np.int16),
        "x2": np.array([[[5, 6], [7, 8]]], dtype=np.int16)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 0D numpy arrays (scalars as arrays), int32
    input_dict = {
        "x1": np.array(7, dtype=np.int32),
        "x2": np.array(24, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large integers, int64
    input_dict = {
        "x1": np.array([100000, 200000], dtype=np.int64),
        "x2": np.array([300000, 400000], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Broadcasting 1D and 2D, int32
    input_dict = {
        "x1": np.array([1, 2, 3], dtype=np.int32),
        "x2": np.array([[4, 5, 6], [7, 8, 9]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 4D numpy arrays, int32
    input_dict = {
        "x1": np.zeros((2, 2, 2, 2), dtype=np.int32) + 3,
        "x2": np.zeros((2, 2, 2, 2), dtype=np.int32) + 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.hypot_3"] = hypot_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.hypot_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.hypot_3'.")


check_valid('jax.numpy.hypot', generated_inputs['jax.numpy.hypot_3'], lib="jax", suffix=3)
