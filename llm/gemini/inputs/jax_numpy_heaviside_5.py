
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def heaviside_inputs():
    list_of_inputs = []

    # Input 1: x1 is zero, x2 is a 1D float32 array
    input_dict = {"x1": 0, "x2": np.array([0.5, 1.0, 0.25], dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: x1 is negative, x2 is a 2D float64 array
    input_dict = {
        "x1": -3,
        "x2": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: x1 is positive, x2 is a 3D float32 array
    input_dict = {
        "x1": 5,
        "x2": np.ones((2, 3, 4), dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: x1 is zero, x2 is a 0D scalar array
    input_dict = {
        "x1": 0,
        "x2": np.array(0.75, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: x1 is negative, x2 is a 1D integer array
    input_dict = {
        "x1": -1,
        "x2": np.array([0, 1, 2], dtype=np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: x1 is positive, x2 is a 4D float32 array
    input_dict = {
        "x1": 10,
        "x2": np.zeros((2, 1, 3, 2), dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: x1 is zero, x2 is an int64 2D array
    input_dict = {
        "x1": 0,
        "x2": np.array([[-5, 0], [5, 10]], dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: x1 is large negative, x2 is float16 array
    input_dict = {
        "x1": -100,
        "x2": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float16),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: x1 is positive, x2 is a boolean array
    input_dict = {
        "x1": 1,
        "x2": np.array([[True, False], [False, True]], dtype=bool),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: x1 is zero, x2 is a 1D array with negative values
    input_dict = {
        "x1": 0,
        "x2": np.array([-0.5, 0.0, 0.5], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: x1 is large positive, x2 is a 5D float64 array
    input_dict = {
        "x1": 42,
        "x2": np.ones((1, 2, 1, 3, 1), dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.heaviside_5"] = heaviside_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.heaviside_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.heaviside_5'.")


check_valid('jax.numpy.heaviside', generated_inputs['jax.numpy.heaviside_5'], lib="jax", suffix=5)
