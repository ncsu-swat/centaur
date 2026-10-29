
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def not_equal_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean array
    input_dict = {
        "x": True,
        "y": np.array([True, False, True, False], dtype=bool),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D boolean array
    input_dict = {
        "x": False,
        "y": np.array([[False, True], [True, False]], dtype=bool),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D int32 array with negative and zero values
    input_dict = {
        "x": True,
        "y": np.array([-1, 0, 1, 2], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D int64 array
    input_dict = {
        "x": False,
        "y": np.array([[0, 1], [-2, 0]], dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D float32 array
    input_dict = {
        "x": True,
        "y": np.array([1.0, 0.0, -1.0, 0.5], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float64 array of zeros
    input_dict = {
        "x": False,
        "y": np.zeros((2, 3, 4), dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 0D scalar array
    input_dict = {
        "x": True,
        "y": np.array(1, dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D uint8 array
    input_dict = {
        "x": False,
        "y": np.ones((2, 2, 2, 2), dtype=np.uint8),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D complex64 array
    input_dict = {
        "x": True,
        "y": np.array([1 + 0j, 0 + 0j, 1 + 2j], dtype=np.complex64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D float32 array containing NaN and Inf
    input_dict = {
        "x": False,
        "y": np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Empty 1D array
    input_dict = {
        "x": True,
        "y": np.empty((0,), dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.not_equal_7"] = not_equal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.not_equal_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.not_equal_7'.")


check_valid('jax.numpy.not_equal', generated_inputs['jax.numpy.not_equal_7'], lib="jax", suffix=7)
