
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def unique_inverse_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array
    x = np.array([3, 4, 1, 3, 1], dtype=np.int32)
    input_dict = {
        "x": x,
        "size": 5,
        "fill_value": -1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float array with negative values
    x = np.array([-1.5, 2.3, -1.5, 0.0, 4.2], dtype=np.float32)
    input_dict = {
        "x": x,
        "size": 3,
        "fill_value": -9.9
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D integer array
    x = np.array([[1, 2], [2, 3], [1, 2]], dtype=np.int32)
    input_dict = {
        "x": x,
        "size": 4,
        "fill_value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D integer array
    x = np.random.randint(0, 5, size=(2, 2, 2)).astype(np.int32)
    input_dict = {
        "x": x,
        "size": 6,
        "fill_value": -1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Large size compared to unique values (padding needed)
    x = np.array([5, 5, 5, 5], dtype=np.int32)
    input_dict = {
        "x": x,
        "size": 3,
        "fill_value": 9.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D float64 array
    x = np.array([0.1, 0.2, 0.1, 0.3, 0.2], dtype=np.float64)
    input_dict = {
        "x": x,
        "size": 4,
        "fill_value": -0.5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array with single element
    x = np.array([42], dtype=np.int32)
    input_dict = {
        "x": x,
        "size": 2,
        "fill_value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D float array
    x = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    input_dict = {
        "x": x,
        "size": 3,
        "fill_value": -100.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 4D array of small scale
    x = np.random.randint(-10, 10, size=(1, 2, 2, 1)).astype(np.int32)
    input_dict = {
        "x": x,
        "size": 5,
        "fill_value": -99.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array of zeros
    x = np.zeros((10,), dtype=np.int32)
    input_dict = {
        "x": x,
        "size": 2,
        "fill_value": -1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.unique_inverse_3"] = unique_inverse_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unique_inverse_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unique_inverse_3'.")


check_valid('jax.numpy.unique_inverse', generated_inputs['jax.numpy.unique_inverse_3'], lib="jax", suffix=3)
