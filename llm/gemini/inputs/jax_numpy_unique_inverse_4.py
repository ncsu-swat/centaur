
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def unique_inverse_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array with positive values, size larger than unique count
    x = np.array([3, 4, 1, 3, 1], dtype=np.int32)
    input_dict = {
        "x": x,
        "size": 5,
        "fill_value": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float array with negative and positive values, size equal to unique count
    x = np.array([-1.5, 2.0, -1.5, 0.0], dtype=np.float32)
    input_dict = {
        "x": x,
        "size": 3,
        "fill_value": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D integer array, size larger than unique count
    x = np.array([[1, 2], [2, 3]], dtype=np.int64)
    input_dict = {
        "x": x,
        "size": 4,
        "fill_value": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D integer array with negative values, size smaller than unique count
    x = np.array([[[1, -1], [2, -2]], [[1, 2], [-1, -2]]], dtype=np.int32)
    input_dict = {
        "x": x,
        "size": 2,
        "fill_value": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D boolean array, size equal to unique count
    x = np.array([True, False, True, False, False], dtype=np.bool_)
    input_dict = {
        "x": x,
        "size": 2,
        "fill_value": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float64 array, size larger than unique count
    x = np.array([[0.5, -0.5, 0.5], [-0.5, 1.5, 1.5]], dtype=np.float64)
    input_dict = {
        "x": x,
        "size": 6,
        "fill_value": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array with descending and negative values, size larger than unique count
    x = np.arange(10, -10, -2, dtype=np.int32)
    input_dict = {
        "x": x,
        "size": 12,
        "fill_value": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array of constant values
    x = np.ones((2, 2, 2), dtype=np.int32)
    input_dict = {
        "x": x,
        "size": 3,
        "fill_value": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D negative integer array
    x = np.array([-10, -20, -10, -30, -20], dtype=np.int32)
    input_dict = {
        "x": x,
        "size": 5,
        "fill_value": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D identical positive values
    x = np.array([5, 5, 5, 5], dtype=np.int32)
    input_dict = {
        "x": x,
        "size": 2,
        "fill_value": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.unique_inverse_4"] = unique_inverse_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unique_inverse_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unique_inverse_4'.")


check_valid('jax.numpy.unique_inverse', generated_inputs['jax.numpy.unique_inverse_4'], lib="jax", suffix=4)
