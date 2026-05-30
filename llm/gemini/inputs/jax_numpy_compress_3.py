
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def compress_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D array, compressing rows
    input_dict = {
        "condition": np.array([True, False, True], dtype=bool),
        "a": np.random.randn(3, 4).astype(np.float32),
        "axis": 0,
        "size": 2,
        "fill_value": np.array(0.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, compressing columns
    input_dict = {
        "condition": np.array([True, False, True, False], dtype=bool),
        "a": np.random.randn(3, 4).astype(np.float32),
        "axis": 1,
        "size": 2,
        "fill_value": np.array(-1.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D integer array condition and 1D integer input
    input_dict = {
        "condition": np.array([1, 0, 1, 1, 0], dtype=np.int32),
        "a": np.arange(5, dtype=np.int32),
        "axis": 0,
        "size": 3,
        "fill_value": np.array(999, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float64 array, compressing along first axis
    input_dict = {
        "condition": np.array([True, True], dtype=bool),
        "a": np.random.randn(2, 3, 2).astype(np.float64),
        "axis": 0,
        "size": 2,
        "fill_value": np.array(0.0, dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int16 array with all-False condition to test padding
    input_dict = {
        "condition": np.array([False, False, False], dtype=bool),
        "a": np.random.randint(-10, 10, size=(2, 3, 4)).astype(np.int16),
        "axis": 1,
        "size": 1,
        "fill_value": np.array(-9, dtype=np.int16)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array, size matches maximum dimension
    input_dict = {
        "condition": np.array([True, True, True, True], dtype=bool),
        "a": np.random.randn(2, 2, 4).astype(np.float32),
        "axis": 2,
        "size": 4,
        "fill_value": np.array(1.5, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative axis index on a 2D float64 array
    input_dict = {
        "condition": np.array([1, 1, 0], dtype=np.int32),
        "a": np.random.randn(3, 2).astype(np.float64),
        "axis": -2,
        "size": 2,
        "fill_value": np.array(0.0, dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D float32 array
    input_dict = {
        "condition": np.array([True, False, True, False, True, True], dtype=bool),
        "a": np.random.randn(6).astype(np.float32),
        "axis": 0,
        "size": 4,
        "fill_value": np.array(np.nan, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D float32 array, compressing last axis
    input_dict = {
        "condition": np.array([True, False], dtype=bool),
        "a": np.random.randn(2, 2, 2, 2).astype(np.float32),
        "axis": 3,
        "size": 1,
        "fill_value": np.array(-0.0, dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger 1D int32 array
    input_dict = {
        "condition": np.array([True] * 10, dtype=bool),
        "a": np.arange(10, dtype=np.int32),
        "axis": 0,
        "size": 10,
        "fill_value": np.array(0, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.compress_3"] = compress_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.compress_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.compress_3'.")


check_valid('jax.numpy.compress', generated_inputs['jax.numpy.compress_3'], lib="jax", suffix=3)
