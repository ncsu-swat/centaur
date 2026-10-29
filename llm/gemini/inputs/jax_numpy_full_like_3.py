
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def full_like_inputs():
    list_of_inputs = []

    # Input 1
    input_dict = {
        "a": np.zeros((3, 3), dtype=np.float32),
        "fill_value": 7,
        "dtype": np.dtype(np.int32),
        "shape": 5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: negative fill_value
    input_dict = {
        "a": np.ones((2, 4), dtype=np.int64),
        "fill_value": -3,
        "dtype": np.dtype(np.float32),
        "shape": 8,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D array, zero fill_value
    input_dict = {
        "a": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        "fill_value": 0,
        "dtype": np.dtype(np.int64),
        "shape": 4,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, boolean dtype
    input_dict = {
        "a": np.zeros((2, 2, 2), dtype=np.int32),
        "fill_value": 1,
        "dtype": np.dtype(np.bool_),
        "shape": 6,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: shape is 0
    input_dict = {
        "a": np.array([], dtype=np.float32),
        "fill_value": 42,
        "dtype": np.dtype(np.float32),
        "shape": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: shape is 1
    input_dict = {
        "a": np.array([[10]], dtype=np.uint8),
        "fill_value": 255,
        "dtype": np.dtype(np.uint8),
        "shape": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: complex dtype
    input_dict = {
        "a": np.arange(6, dtype=np.float32).reshape(2, 3),
        "fill_value": -10,
        "dtype": np.dtype(np.complex64),
        "shape": 3,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D array, int16 dtype
    input_dict = {
        "a": np.ones((1, 2, 1, 3), dtype=np.int16),
        "fill_value": 100,
        "dtype": np.dtype(np.int16),
        "shape": 12,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 0D scalar array
    input_dict = {
        "a": np.array(5.5, dtype=np.float32),
        "fill_value": -1,
        "dtype": np.dtype(np.int8),
        "shape": 2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: large shape
    input_dict = {
        "a": np.zeros(10, dtype=np.float64),
        "fill_value": 999,
        "dtype": np.dtype(np.float64),
        "shape": 100,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: uint32 dtype
    input_dict = {
        "a": np.array([True, False, True], dtype=np.bool_),
        "fill_value": 12345,
        "dtype": np.dtype(np.uint32),
        "shape": 7,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.full_like_3"] = full_like_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.full_like_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.full_like_3'.")


check_valid('jax.numpy.full_like', generated_inputs['jax.numpy.full_like_3'], lib="jax", suffix=3)
