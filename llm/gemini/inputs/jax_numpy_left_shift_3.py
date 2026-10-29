
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def left_shift_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array
    input_dict = {
        "x": 1,
        "y": np.array([1, 2, 3, 4], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int64 array
    input_dict = {
        "x": 4,
        "y": np.array([[1, 2], [3, 0]], dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative integer for x, 1D int32 array
    input_dict = {
        "x": -2,
        "y": np.array([0, 1, 2, 3], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D int32 array with zeros
    input_dict = {
        "x": 0,
        "y": np.zeros((3, 3, 3), dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int32 array with small shift amounts
    input_dict = {
        "x": 15,
        "y": np.array([[[1, 2], [3, 4]], [[0, 1], [2, 3]]], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative integer for x, 1D int64 array
    input_dict = {
        "x": -10,
        "y": np.array([1, 2, 3], dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 0-D scalar tensor for y
    input_dict = {
        "x": 7,
        "y": np.array(2, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D column vector with int32
    input_dict = {
        "x": 3,
        "y": np.ones((5, 1), dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D int64 array from arange
    input_dict = {
        "x": 5,
        "y": np.arange(10, dtype=np.int64).reshape(2, 5),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D int32 array
    input_dict = {
        "x": -1,
        "y": np.array([[[[0, 1], [2, 3]]]], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Empty 1D array
    input_dict = {
        "x": 42,
        "y": np.array([], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.left_shift_3"] = left_shift_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.left_shift_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.left_shift_3'.")


check_valid('jax.numpy.left_shift', generated_inputs['jax.numpy.left_shift_3'], lib="jax", suffix=3)
