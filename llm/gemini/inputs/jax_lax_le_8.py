
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def lax_le_inputs():
    list_of_inputs = []

    # Input 1: 1D array with positive integers
    input_dict = {
        "x": 3,
        "y": np.array([1, 2, 3, 4, 5], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array with negative and positive values
    input_dict = {
        "x": 0,
        "y": np.array([[-2, -1, 0], [1, 2, 3]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative integer scalar x and 1D array
    input_dict = {
        "x": -5,
        "y": np.array([-10, -5, 0, 5], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 0-dimensional (scalar) array
    input_dict = {
        "x": 10,
        "y": np.array(10, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array
    input_dict = {
        "x": 1,
        "y": np.zeros((2, 3, 4), dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array with large integers
    input_dict = {
        "x": 100,
        "y": np.arange(16, dtype=np.int32).reshape((2, 2, 2, 2)) * 10
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty 1D array
    input_dict = {
        "x": 0,
        "y": np.array([], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array with single row
    input_dict = {
        "x": -1,
        "y": np.array([[-5, -3, -1, 1, 3]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array with shape (1, 1, 1)
    input_dict = {
        "x": 42,
        "y": np.array([[[42]]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 5D array
    input_dict = {
        "x": 2,
        "y": np.ones((1, 2, 1, 3, 1), dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D column vector
    input_dict = {
        "x": 7,
        "y": np.array([[3], [7], [12]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.le_8"] = lax_le_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.le_8' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.le_8'.")


check_valid('jax.lax.le', generated_inputs['jax.lax.le_8'], lib="jax", suffix=8)
