
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def flatnonzero_inputs():
    list_of_inputs = []

    # Input 1: Simple 1D array
    a = np.array([0, 1, 2, 0, 3, 0]).astype(np.int32)
    input_dict = {
        "a": a,
        "size": 3,
        "fill_value": -1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array with some zeros
    a = np.array([[1, 0, 3], [0, 5, 0]]).astype(np.int32)
    input_dict = {
        "a": a,
        "size": 5,
        "fill_value": -1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float array and smaller size (truncation)
    a = np.array([0.5, 0.0, -1.2, 3.4, 0.0]).astype(np.float32)
    input_dict = {
        "a": a,
        "size": 2,
        "fill_value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array
    a = np.array([[[0, 1], [2, 0]], [[0, 0], [3, 4]]]).astype(np.int32)
    input_dict = {
        "a": a,
        "size": 6,
        "fill_value": -99.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: All zeros array
    a = np.zeros((3, 3), dtype=np.int32)
    input_dict = {
        "a": a,
        "size": 4,
        "fill_value": -1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Large 1D array
    a = np.random.choice([0, 1], size=(100,)).astype(np.int32)
    input_dict = {
        "a": a,
        "size": 50,
        "fill_value": -1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Boolean array acting as integer
    a = np.array([True, False, True, True, False])
    input_dict = {
        "a": a,
        "size": 5,
        "fill_value": -1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Float64 array with positive and negative values
    a = np.array([-1.5, 0.0, 2.5, -3.5]).astype(np.float64)
    input_dict = {
        "a": a,
        "size": 3,
        "fill_value": -2.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D array
    a = np.random.choice([0, 5], size=(2, 2, 2, 2)).astype(np.int32)
    input_dict = {
        "a": a,
        "size": 10,
        "fill_value": -1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Array with size exact match
    a = np.array([1, 2, 3, 4]).astype(np.int32)
    input_dict = {
        "a": a,
        "size": 4,
        "fill_value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.flatnonzero_3"] = flatnonzero_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.flatnonzero_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.flatnonzero_3'.")


check_valid('jax.numpy.flatnonzero', generated_inputs['jax.numpy.flatnonzero_3'], lib="jax", suffix=3)
