
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def union1d_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D int32 arrays with padding
    input_dict = {
        "ar1": np.array([1, 2, 3, 4], dtype=np.int32),
        "ar2": np.array([3, 4, 5, 6], dtype=np.int32),
        "size": 8,
        "fill_value": np.array(0, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D int64 arrays with negative numbers and truncation
    input_dict = {
        "ar1": np.array([-10, -5, 0, 5], dtype=np.int64),
        "ar2": np.array([-5, 0, 10, 15], dtype=np.int64),
        "size": 3,
        "fill_value": np.array(-999, dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float32 arrays with exact size
    input_dict = {
        "ar1": np.array([1.5, 2.5, 3.5], dtype=np.float32),
        "ar2": np.array([3.5, 4.5, 5.5], dtype=np.float32),
        "size": 5,
        "fill_value": np.array(np.nan, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Multidimensional 2D float64 arrays
    input_dict = {
        "ar1": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        "ar2": np.array([[3.0, 5.0], [6.0, 7.0]], dtype=np.float64),
        "size": 10,
        "fill_value": np.array(-1.0, dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Boolean arrays
    input_dict = {
        "ar1": np.array([True, False, True], dtype=np.bool_),
        "ar2": np.array([False, False], dtype=np.bool_),
        "size": 4,
        "fill_value": np.array(False, dtype=np.bool_),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Unsigned integer uint8 arrays
    input_dict = {
        "ar1": np.array([10, 20, 30, 40], dtype=np.uint8),
        "ar2": np.array([20, 30, 50, 60], dtype=np.uint8),
        "size": 6,
        "fill_value": np.array(0, dtype=np.uint8),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty arrays
    input_dict = {
        "ar1": np.array([], dtype=np.int32),
        "ar2": np.array([], dtype=np.int32),
        "size": 3,
        "fill_value": np.array(-1, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: One empty array and one non-empty 1D array
    input_dict = {
        "ar1": np.array([], dtype=np.float32),
        "ar2": np.array([1.1, 2.2, 3.3], dtype=np.float32),
        "size": 5,
        "fill_value": np.array(0.0, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D int16 arrays with duplicate elements
    input_dict = {
        "ar1": np.ones((2, 2, 2), dtype=np.int16),
        "ar2": np.zeros((1, 3, 2), dtype=np.int16),
        "size": 4,
        "fill_value": np.array(-1, dtype=np.int16),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Size 0 test
    input_dict = {
        "ar1": np.array([100, 200, 300], dtype=np.int32),
        "ar2": np.array([300, 400, 500], dtype=np.int32),
        "size": 0,
        "fill_value": np.array(0, dtype=np.int32),
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
