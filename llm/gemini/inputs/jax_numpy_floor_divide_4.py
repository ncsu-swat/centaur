
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def floor_divide_inputs():
    list_of_inputs = []

    # Input 1: positive integer and 1D integer array
    input_dict = {
        "x1": 10,
        "x2": np.array([1, 2, 3, 4, 5], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: negative integer and 1D integer array
    input_dict = {
        "x1": -15,
        "x2": np.array([2, 3, 4, 5, 6], dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: zero integer and 2D integer array with mixed signs
    input_dict = {
        "x1": 0,
        "x2": np.array([[1, -2], [3, -4]], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: positive integer and 3D float array
    input_dict = {
        "x1": 100,
        "x2": np.array([[[1.5, 2.5], [3.5, 4.5]]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: negative integer and 1D array with positive and negative integers
    input_dict = {
        "x1": -7,
        "x2": np.array([-3, -2, -1, 1, 2, 3], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: positive integer and 4D int16 array
    input_dict = {
        "x1": 42,
        "x2": np.array([[[[2, 3]], [[4, 5]]]], dtype=np.int16),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: positive integer and 1D float64 array
    input_dict = {
        "x1": 1,
        "x2": np.array([0.5, 0.25, 0.125], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: negative integer and 2D float32 array
    input_dict = {
        "x1": -50,
        "x2": np.array([[-10.0, 5.0], [-2.5, 20.0]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: large positive integer and 1D int64 array
    input_dict = {
        "x1": 123456789,
        "x2": np.array([10, 100, 1000, 10000], dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: integer and 0D array (scalar tensor)
    input_dict = {
        "x1": 8,
        "x2": np.array(4, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: positive integer and 1D unsigned integer array
    input_dict = {
        "x1": 255,
        "x2": np.array([1, 2, 4, 8, 16], dtype=np.uint8),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.floor_divide_4"] = floor_divide_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.floor_divide_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.floor_divide_4'.")


check_valid('jax.numpy.floor_divide', generated_inputs['jax.numpy.floor_divide_4'], lib="jax", suffix=4)
