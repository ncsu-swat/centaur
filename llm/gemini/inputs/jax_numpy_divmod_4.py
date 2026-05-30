
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def divmod_inputs():
    list_of_inputs = []

    # Input 1: positive integer, 1D int32 array
    input_dict = {
        "x1": 10,
        "x2": np.array([3, 4, 7], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: negative integer, 1D int32 array
    input_dict = {
        "x1": -5,
        "x2": np.array([1, 2, 3, 4], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: zero integer, 1D int32 array with negative values
    input_dict = {
        "x1": 0,
        "x2": np.array([-3, -2, -1], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: positive integer, 2D int32 array
    input_dict = {
        "x1": 100,
        "x2": np.array([[3, 7, 9], [11, 13, 17]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: negative integer, 2D int64 array with negative values
    input_dict = {
        "x1": -50,
        "x2": np.array([[-15, -20], [-25, -30]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: positive integer, 2D float32 array
    input_dict = {
        "x1": 12,
        "x2": np.array([[1.5, 2.0], [3.5, 4.0]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: positive integer, 3D int32 array
    input_dict = {
        "x1": 7,
        "x2": np.ones((2, 2, 2), dtype=np.int32) * 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: negative integer, 1D float64 array
    input_dict = {
        "x1": -15,
        "x2": np.array([2.5, 3.5, 4.5], dtype=np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: large positive integer, 3D int16 array
    input_dict = {
        "x1": 1000,
        "x2": np.array([[[5, 10], [15, 20]]], dtype=np.int16)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: positive integer, 1-element 1D int32 array
    input_dict = {
        "x1": 42,
        "x2": np.array([5], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.divmod_4"] = divmod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.divmod_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.divmod_4'.")


check_valid('jax.numpy.divmod', generated_inputs['jax.numpy.divmod_4'], lib="jax", suffix=4)
