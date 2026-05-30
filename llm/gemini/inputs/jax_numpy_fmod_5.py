
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def fmod_inputs():
    list_of_inputs = []

    # Input 1: positive integer, 1D int32 array
    input_dict = {
        "x1": 5,
        "x2": np.array([2, 3, 4], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: negative integer, 2D float32 array with positive/negative values
    input_dict = {
        "x1": -10,
        "x2": np.array([[3.0, -3.0], [4.0, -4.0]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: zero integer, 1D int64 array
    input_dict = {
        "x1": 0,
        "x2": np.array([1, 2, 3], dtype=np.int64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: positive integer, random positive 1D int32 array (no zeros)
    input_dict = {
        "x1": 100,
        "x2": np.random.randint(1, 10, size=(5,)).astype(np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: negative integer, random 2D float64 array
    input_dict = {
        "x1": -7,
        "x2": np.random.uniform(1.0, 5.0, size=(3, 3)).astype(np.float64)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: positive integer, 3D int32 array
    input_dict = {
        "x1": 42,
        "x2": np.array([[[2, 5], [7, 9]], [[11, 13], [17, 19]]], dtype=np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: positive integer, 1D float32 array with decimal values
    input_dict = {
        "x1": 12,
        "x2": np.array([2.5, 3.5, 4.5], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: large negative integer, negative 2D int32 array
    input_dict = {
        "x1": -1000,
        "x2": np.random.randint(-50, -10, size=(2, 4)).astype(np.int32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: small positive integer, 2D float32 array with values less than 1
    input_dict = {
        "x1": 1,
        "x2": np.array([[0.1, 0.2], [0.5, 0.8]], dtype=np.float32)
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: large positive integer, 3D int64 array
    input_dict = {
        "x1": 12345,
        "x2": np.ones((2, 2, 2), dtype=np.int64) * 10
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.fmod_5"] = fmod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fmod_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fmod_5'.")


check_valid('jax.numpy.fmod', generated_inputs['jax.numpy.fmod_5'], lib="jax", suffix=5)
