
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def union1d_inputs():
    list_of_inputs = []

    # Input 1: Basic union with size fitting perfectly
    ar1 = np.array([1, 2, 3, 4], dtype=np.int32)
    ar2 = np.array([3, 4, 5, 6], dtype=np.int32)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "size": 6,
        "fill_value": np.array(0, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Truncated output (size smaller than unique union size)
    ar1 = np.array([-1, -2, -3], dtype=np.int32)
    ar2 = np.array([-3, -4, -5], dtype=np.int32)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "size": 3,
        "fill_value": np.array(-9, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Padded output with duplicates in input
    ar1 = np.array([1, 1, 2, 2], dtype=np.int32)
    ar2 = np.array([2, 2, 3, 3], dtype=np.int32)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "size": 5,
        "fill_value": np.array(9, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Float32 tensors
    ar1 = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    ar2 = np.array([3.5, 4.5, 5.5], dtype=np.float32)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "size": 5,
        "fill_value": np.array(-1.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Float64 tensors with padding
    ar1 = np.array([0.1, 0.2], dtype=np.float64)
    ar2 = np.array([0.2, 0.3], dtype=np.float64)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "size": 4,
        "fill_value": np.array(-0.5, dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Standard Int32 case
    ar1 = np.array([100, 200], dtype=np.int32)
    ar2 = np.array([200, 300], dtype=np.int32)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "size": 3,
        "fill_value": np.array(0, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Int64 tensors with large values
    ar1 = np.array([1000000, 2000000], dtype=np.int64)
    ar2 = np.array([2000000, 3000000], dtype=np.int64)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "size": 4,
        "fill_value": np.array(0, dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Another standard Int32 case
    ar1 = np.array([10, 20, 30], dtype=np.int32)
    ar2 = np.array([30, 40, 50], dtype=np.int32)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "size": 6,
        "fill_value": np.array(99, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty tensors
    ar1 = np.array([], dtype=np.int32)
    ar2 = np.array([], dtype=np.int32)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "size": 3,
        "fill_value": np.array(-1, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Multi-dimensional tensors (which will be flattened)
    ar1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    ar2 = np.array([[3, 4], [5, 6]], dtype=np.int32)
    input_dict = {
        "ar1": ar1,
        "ar2": ar2,
        "size": 8,
        "fill_value": np.array(-99, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.union1d_1"] = union1d_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.union1d_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.union1d_1'.")


check_valid('jax.numpy.union1d', generated_inputs['jax.numpy.union1d_1'], lib="jax", suffix=1)
