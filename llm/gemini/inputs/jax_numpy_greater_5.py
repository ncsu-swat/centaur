
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def greater_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array, positive integer scalar
    input_dict = {
        "x": 5,
        "y": np.array([1, 5, 10, -2, 0], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float32 array, zero scalar
    input_dict = {
        "x": 0,
        "y": np.array([[-1.5, 0.0], [2.3, -0.1]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D negative integer scalar with int64 array
    input_dict = {
        "x": -3,
        "y": np.array([-5, -3, -1, 0, 2], dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float64 array
    input_dict = {
        "x": 10,
        "y": np.arange(24, dtype=np.float64).reshape((2, 3, 4)),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 0D array (scalar tensor)
    input_dict = {
        "x": 42,
        "y": np.array(42, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D uint8 array
    input_dict = {
        "x": 128,
        "y": np.random.randint(0, 256, size=(2, 2, 3, 3), dtype=np.uint8),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty 1D array
    input_dict = {
        "x": 1,
        "y": np.array([], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D boolean array
    input_dict = {
        "x": 1,
        "y": np.array([[True, False], [False, True]], dtype=np.bool_),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large integer with int64 array
    input_dict = {
        "x": 1000000,
        "y": np.array([999999, 1000000, 1000001], dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array with special float values (inf, -inf, nan)
    input_dict = {
        "x": 0,
        "y": np.array([-np.inf, -1.0, 0.0, 1.0, np.inf, np.nan], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 5D array
    input_dict = {
        "x": -10,
        "y": np.ones((1, 2, 1, 3, 2), dtype=np.int16),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.greater_5"] = greater_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.greater_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.greater_5'.")


check_valid('jax.numpy.greater', generated_inputs['jax.numpy.greater_5'], lib="jax", suffix=5)
