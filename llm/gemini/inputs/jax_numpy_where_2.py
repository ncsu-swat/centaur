
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def where_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D boolean array, small size
    condition = np.array([True, False, True, False, True], dtype=bool)
    input_dict = {
        "condition": condition,
        "size": 5,
        "fill_value": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D boolean array, size larger than True elements count
    condition = np.array([[True, False], [False, True]], dtype=bool)
    input_dict = {
        "condition": condition,
        "size": 4,
        "fill_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D boolean array, size smaller than True elements count (truncation case)
    condition = np.array([True, True, True, True, True], dtype=bool)
    input_dict = {
        "condition": condition,
        "size": 2,
        "fill_value": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D random boolean array
    condition = np.random.choice([True, False], size=(2, 2, 2))
    input_dict = {
        "condition": condition,
        "size": 8,
        "fill_value": -999
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D boolean array with no True elements
    condition = np.array([False, False, False, False], dtype=bool)
    input_dict = {
        "condition": condition,
        "size": 4,
        "fill_value": 99
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D random boolean array with negative fill value
    condition = np.random.choice([True, False], size=(5, 5))
    input_dict = {
        "condition": condition,
        "size": 15,
        "fill_value": -2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D boolean array
    condition = np.random.choice([True, False], size=(2, 1, 3, 2))
    input_dict = {
        "condition": condition,
        "size": 12,
        "fill_value": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D boolean array with a very large size parameter
    condition = np.array([True], dtype=bool)
    input_dict = {
        "condition": condition,
        "size": 100,
        "fill_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D boolean array, all elements are True
    condition = np.ones((3, 3), dtype=bool)
    input_dict = {
        "condition": condition,
        "size": 15,
        "fill_value": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D boolean array, size matches exact number of elements
    condition = np.array([True, False, True, False], dtype=bool)
    input_dict = {
        "condition": condition,
        "size": 4,
        "fill_value": -5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.where_2"] = where_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.where_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.where_2'.")


check_valid('jax.numpy.where', generated_inputs['jax.numpy.where_2'], lib="jax", suffix=2)
