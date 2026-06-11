
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def clamp_inputs():
    list_of_inputs = []

    # Input 1: 1D array
    min_val = -5
    max_val = 5
    x = np.array([-10, -2, 0, 3, 8], dtype=int)
    input_dict = {"min": min_val, "x": x, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array
    min_val = 2
    max_val = 8
    x = np.array([[1, 2, 3], [7, 8, 9]], dtype=int)
    input_dict = {"min": min_val, "x": x, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array with negative range
    min_val = -20
    max_val = -10
    x = np.array([[-15, -12], [-25, -5]], dtype=int)
    input_dict = {"min": min_val, "x": x, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D array with wide range
    min_val = -100
    max_val = 100
    x = np.array([-200, -50, 0, 50, 200], dtype=int)
    input_dict = {"min": min_val, "x": x, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array
    min_val = 0
    max_val = 10
    x = np.array([[[1, 2], [3, 4]], [[5, 6], [11, 12]]], dtype=int)
    input_dict = {"min": min_val, "x": x, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array, small range
    min_val = -3
    max_val = 3
    x = np.array([[-5, -1], [0, 4]], dtype=int)
    input_dict = {"min": min_val, "x": x, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array, min/max equal
    min_val = 5
    max_val = 5
    x = np.array([[2, 5], [8, 5]], dtype=int)
    input_dict = {"min": min_val, "x": x, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array, large scale
    min_val = -1000
    max_val = 1000
    x = np.array([-2000, 0, 2000], dtype=int)
    input_dict = {"min": min_val, "x": x, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D array
    min_val = -10
    max_val = 10
    x = np.array([[[[[-15, 0], [15, 5]]]]], dtype=int)
    input_dict = {"min": min_val, "x": x, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array, tight bounds
    min_val = 1
    max_val = 2
    x = np.array([0, 1, 2, 3], dtype=int)
    input_dict = {"min": min_val, "x": x, "max": max_val}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.lax.clamp_3"] = clamp_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.lax.clamp_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.lax.clamp_3'.")


check_valid('jax.lax.clamp', generated_inputs['jax.lax.clamp_3'], lib="jax", suffix=3)
