
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def equal_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array
    input_dict = {
        "x": 5,
        "y": np.array([5, 3, 5, 2], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D int32 array of zeros
    input_dict = {
        "x": 0,
        "y": np.zeros((3, 3), dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D int64 array with negative values
    input_dict = {
        "x": -1,
        "y": np.array([[-1, 2], [3, -1]], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D random int32 array
    input_dict = {
        "x": 10,
        "y": np.random.randint(0, 20, size=(2, 3, 4), dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D float32 array
    input_dict = {
        "x": 1,
        "y": np.ones((5,), dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D int32 array with 2 elements
    input_dict = {
        "x": 42,
        "y": np.array([42, 0], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D int32 array
    input_dict = {
        "x": -5,
        "y": np.array([[[-5, 5], [1, -5]], [[-5, 0], [0, 0]]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D float64 array
    input_dict = {
        "x": 0,
        "y": np.random.randn(2, 2).astype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D int32 array of shape 1x5 filled with integer
    input_dict = {
        "x": 3,
        "y": np.full((1, 5), 3, dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D int64 array
    input_dict = {
        "x": 100,
        "y": np.array([100], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.equal_5"] = equal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.equal_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.equal_5'.")


check_valid('jax.numpy.equal', generated_inputs['jax.numpy.equal_5'], lib="jax", suffix=5)
