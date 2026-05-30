
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_greater_inputs():
    list_of_inputs = []

    # Input 1: Basic Python integers
    input_dict = {
        "x": 5,
        "y": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Python integers with negative values
    input_dict = {
        "x": -10,
        "y": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D numpy integer arrays of same shape (int32)
    input_dict = {
        "x": np.array([1, -5, 10], dtype=np.int32),
        "y": np.array([2, -5, 8], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D numpy integer arrays (int64)
    input_dict = {
        "x": np.array([[10, -5], [4, 2]], dtype=np.int64),
        "y": np.array([[5, -10], [4, 1]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: numpy integer scalars
    input_dict = {
        "x": np.int32(-15),
        "y": np.int32(-20)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Broadcasting 1D array with 2D array (int16)
    input_dict = {
        "x": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16),
        "y": np.array([2, 2, 2], dtype=np.int16)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D numpy arrays (int8)
    input_dict = {
        "x": np.array([[[1, 2], [3, 4]]], dtype=np.int8),
        "y": np.array([[[2, 1], [4, 3]]], dtype=np.int8)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Scalar array broadcasting
    input_dict = {
        "x": np.array([10, 20, 30], dtype=np.int32),
        "y": np.array([15], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large integer values (int64)
    input_dict = {
        "x": np.array([9223372036854775807, -9223372036854775808], dtype=np.int64),
        "y": np.array([0, 0], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Multi-dimensional random integer arrays
    input_dict = {
        "x": np.random.randint(-100, 100, size=(2, 3, 4), dtype=np.int32),
        "y": np.random.randint(-100, 100, size=(2, 3, 4), dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.greater_7"] = jax_numpy_greater_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.greater_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.greater_7'.")


check_valid('jax.numpy.greater', generated_inputs['jax.numpy.greater_7'], lib="jax", suffix=7)
