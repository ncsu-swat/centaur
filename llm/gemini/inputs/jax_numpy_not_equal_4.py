
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def not_equal_inputs():
    list_of_inputs = []

    # Input 1: 0D scalar array with positive float
    input_dict = {"x": 3.14, "y": np.array(3.14, dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array with negative float
    input_dict = {
        "x": -2.5,
        "y": np.array([-2.5, 0.0, 2.5, -5.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array of integers
    input_dict = {
        "x": 0.0,
        "y": np.array([[0, 1, -1], [2, 0, 3]], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array of booleans
    input_dict = {
        "x": 1.0,
        "y": np.array([[True, False], [False, True]], dtype=np.bool_),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array with float16
    input_dict = {"x": 0.5, "y": np.ones((2, 3, 4), dtype=np.float16) * 0.5}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D array with random normal values
    input_dict = {
        "x": 1.23,
        "y": np.random.randn(2, 2, 3, 3).astype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Array with NaN and Inf
    input_dict = {
        "x": float("nan"),
        "y": np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Array with infinity
    input_dict = {
        "x": float("inf"),
        "y": np.array([[np.inf, -np.inf], [1.0, np.inf]], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D complex array compared to float
    input_dict = {
        "x": 2.0,
        "y": np.array([2.0 + 0.0j, 2.0 + 1.0j, 0.0 + 0.0j], dtype=np.complex64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Empty array
    input_dict = {"x": 42.0, "y": np.empty((0, 5), dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: High-dimensional array
    input_dict = {
        "x": -0.0,
        "y": np.zeros((1, 2, 1, 3, 2), dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.not_equal_4"] = not_equal_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.not_equal_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.not_equal_4'.")


check_valid('jax.numpy.not_equal', generated_inputs['jax.numpy.not_equal_4'], lib="jax", suffix=4)
