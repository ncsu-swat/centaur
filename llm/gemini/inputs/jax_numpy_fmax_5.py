
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def fmax_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array with positive integer scalar
    input_dict = {
        "x1": 5,
        "x2": np.array([1, 7, 9, 4], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array with negative integer scalar
    input_dict = {
        "x1": -3,
        "x2": np.array([[-5, -2], [0, -10]], dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D float array with zero scalar
    input_dict = {
        "x1": 0,
        "x2": np.array([-1.5, 0.0, 2.3, -0.1], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float array containing NaN and Inf
    input_dict = {
        "x1": 2,
        "x2": np.array(
            [[np.nan, np.inf], [-np.inf, 1.0]], dtype=np.float64
        ),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float32 array
    input_dict = {
        "x1": 10,
        "x2": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 0D scalar array
    input_dict = {
        "x1": -1,
        "x2": np.array(-5, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D array with negative integer scalar
    input_dict = {
        "x1": -10,
        "x2": np.zeros((2, 2, 2, 2), dtype=np.int16),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D unsigned integer array
    input_dict = {
        "x1": 100,
        "x2": np.array([50, 150, 200, 25], dtype=np.uint8),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Empty array
    input_dict = {
        "x1": 42,
        "x2": np.empty((0, 5), dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Array with all NaNs to test NaN handling with integer
    input_dict = {
        "x1": 7,
        "x2": np.array([np.nan, np.nan, np.nan], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Large integer scalar with float64 array
    input_dict = {
        "x1": 1000000,
        "x2": np.array([-1e7, 1e5, 1e6, 1e7], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.fmax_5"] = fmax_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.fmax_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.fmax_5'.")


check_valid('jax.numpy.fmax', generated_inputs['jax.numpy.fmax_5'], lib="jax", suffix=5)
