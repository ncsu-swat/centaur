
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def floor_divide_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array divisor
    input_dict = {
        "x1": 10.0,
        "x2": np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Negative float dividend, 1D int32 array divisor
    input_dict = {"x1": -15.5, "x2": np.array([1, 2, 4, 5], dtype=np.int32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Zero dividend, 2D float64 array divisor with mixed signs
    input_dict = {
        "x1": 0.0,
        "x2": np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32 array divisor
    input_dict = {
        "x1": 100.25,
        "x2": np.full((2, 3, 4), 3.5, dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Negative float dividend, 2D float32 array divisor
    input_dict = {
        "x1": -7.0,
        "x2": np.array([[-1.5, 2.5, -3.5]], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 0D scalar array divisor
    input_dict = {"x1": 3.14159, "x2": np.array(2.0, dtype=np.float32)}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D float64 array divisor with positive values
    input_dict = {
        "x1": 50.0,
        "x2": np.array([1.5, 3.2, 7.8, 12.4, 25.0], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Negative float dividend, 3D int64 array divisor
    input_dict = {
        "x1": -42.0,
        "x2": np.array([[[2], [4]], [[-3], [-6]]], dtype=np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large float dividend, 1D float32 array divisor
    input_dict = {
        "x1": 1.0e5,
        "x2": np.array([10.0, 100.0, 1000.0], dtype=np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Fractional values for dividend and float64 divisor
    input_dict = {
        "x1": -0.5,
        "x2": np.array([-0.2, 0.2, -0.4, 0.4], dtype=np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.floor_divide_5"] = floor_divide_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.floor_divide_5' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.floor_divide_5'.")


check_valid('jax.numpy.floor_divide', generated_inputs['jax.numpy.floor_divide_5'], lib="jax", suffix=5)
