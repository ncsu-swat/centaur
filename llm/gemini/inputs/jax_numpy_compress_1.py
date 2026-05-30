
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def compress_inputs():
    list_of_inputs = []

    # Input 1, valid — 2D array, compressing along axis 0, size <= axis size
    input_dict = {
        "condition": np.array([True, False, True], dtype=bool),
        "a": np.arange(12, dtype=np.int32).reshape(3, 4),
        "axis": 0,
        "size": 2,
        "fill_value": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid — 2D array, compressing along axis 1, size <= axis size
    input_dict = {
        "condition": np.array([False, True, True, False], dtype=bool),
        "a": np.arange(12, dtype=np.float32).reshape(3, 4),
        "axis": 1,
        "size": 2,
        "fill_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3, valid — 3D array, compressing along axis 0
    input_dict = {
        "condition": np.array([True, True], dtype=bool),
        "a": np.arange(24, dtype=np.int32).reshape(2, 3, 4),
        "axis": 0,
        "size": 1,
        "fill_value": 999
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4, valid — 1D array
    input_dict = {
        "condition": np.array([True, False, True, False, True], dtype=bool),
        "a": np.arange(5, dtype=np.int32),
        "axis": 0,
        "size": 3,
        "fill_value": -99
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5, valid — 3D array, compressing along axis 1, size <= axis 1 size (3)
    input_dict = {
        "condition": np.array([True, True, True], dtype=bool),
        "a": np.arange(24, dtype=np.int32).reshape(2, 3, 4),
        "axis": 1,
        "size": 2,
        "fill_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6, valid — compressing along axis 1 with exact size matching
    input_dict = {
        "condition": np.array([False, False, True], dtype=bool),
        "a": np.arange(12, dtype=np.int32).reshape(4, 3),
        "axis": 1,
        "size": 1,
        "fill_value": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7, valid — size truncation when condition has more Trues than size
    input_dict = {
        "condition": np.array([True, True, False, True], dtype=bool),
        "a": np.arange(8, dtype=np.int32).reshape(2, 4),
        "axis": 1,
        "size": 2,
        "fill_value": -5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8, valid — condition size is equal to axis size, smaller static size
    input_dict = {
        "condition": np.array([True, False], dtype=bool),
        "a": np.array([[10, 20], [30, 40]], dtype=np.int32),
        "axis": 0,
        "size": 1,
        "fill_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9, valid — compressing along negative axis (-1)
    input_dict = {
        "condition": np.array([True, False, True, True], dtype=bool),
        "a": np.arange(4, dtype=np.int32),
        "axis": -1,
        "size": 2,
        "fill_value": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10, valid — 2D array, single element output
    input_dict = {
        "condition": np.array([False, True, False], dtype=bool),
        "a": np.arange(9, dtype=np.int32).reshape(3, 3),
        "axis": 0,
        "size": 1,
        "fill_value": 42
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.compress_1"] = compress_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.compress_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.compress_1'.")


check_valid('jax.numpy.compress', generated_inputs['jax.numpy.compress_1'], lib="jax", suffix=1)
