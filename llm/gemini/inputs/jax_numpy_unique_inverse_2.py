
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def unique_inverse_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array, size matches number of uniques
    input_dict = {
        "x": np.array([3, 4, 1, 3, 1], dtype=np.int32),
        "size": 3,
        "fill_value": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D integer array, size is larger than uniques, using fill_value
    input_dict = {
        "x": np.array([3, 4, 1, 3, 1], dtype=np.int32),
        "size": 5,
        "fill_value": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D integer array with positive values
    input_dict = {
        "x": np.array([[1, 2], [2, 3]], dtype=np.int32),
        "size": 2,
        "fill_value": 9
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 1D negative integers with large fill value
    input_dict = {
        "x": np.array([-10, -20, -10, 0, 10, 20], dtype=np.int32),
        "size": 4,
        "fill_value": -999
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D integer array
    input_dict = {
        "x": np.array([[[1, 2], [3, 4]], [[1, 2], [5, 6]]], dtype=np.int32),
        "size": 10,
        "fill_value": 99
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Float array with integer fill_value
    input_dict = {
        "x": np.array([1.5, 2.5, 1.5, 3.5], dtype=np.float32),
        "size": 5,
        "fill_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Single element array
    input_dict = {
        "x": np.array([100], dtype=np.int64),
        "size": 1,
        "fill_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Array of identical zeros
    input_dict = {
        "x": np.zeros((5,), dtype=np.int32),
        "size": 2,
        "fill_value": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Array with repeating values and size smaller than actual unique count
    input_dict = {
        "x": np.array([5, 10, 15, 20, 25], dtype=np.int32),
        "size": 3,
        "fill_value": 10
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D integer array with a large size constraint
    input_dict = {
        "x": np.ones((2, 2, 2, 2), dtype=np.int32),
        "size": 8,
        "fill_value": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.unique_inverse_2"] = unique_inverse_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unique_inverse_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unique_inverse_2'.")


check_valid('jax.numpy.unique_inverse', generated_inputs['jax.numpy.unique_inverse_2'], lib="jax", suffix=2)
