
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def flatnonzero_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array
    a = np.array([0, 3, 0, 4, 0, 5], dtype=np.int32)
    size = 5
    fill_value = (-1,)
    input_dict = {"a": a, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float array
    a = np.array([[1.0, 0.0, 2.0], [0.0, 0.0, 3.0]], dtype=np.float32)
    size = 4
    fill_value = (-1,)
    input_dict = {"a": a, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D int16 array with negative values
    a = np.array([[[0, -1], [2, 0]], [[0, 0], [3, -4]]], dtype=np.int16)
    size = 6
    fill_value = (-999,)
    input_dict = {"a": a, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Boolean array
    a = np.array([True, False, True, False, True], dtype=bool)
    size = 5
    fill_value = (0,)
    input_dict = {"a": a, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 1D float64 array, all zeros
    a = np.zeros(10, dtype=np.float64)
    size = 3
    fill_value = (-1,)
    input_dict = {"a": a, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D int64 array with size smaller than nonzero count
    a = np.ones((3, 3), dtype=np.int64)
    size = 4
    fill_value = (-1,)
    input_dict = {"a": a, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D array
    a = np.zeros((2, 2, 2, 2), dtype=np.int32)
    a[0, 1, 0, 1] = 5
    a[1, 0, 1, 0] = 10
    size = 3
    fill_value = (-1,)
    input_dict = {"a": a, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D array, large size with custom fill value
    a = np.array([0, 0, 1, 0, 0, 2, 0, 3], dtype=np.int32)
    size = 10
    fill_value = (-5,)
    input_dict = {"a": a, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array, size exactly equal to nonzero count
    a = np.array([[0, 1], [2, 0]], dtype=np.int32)
    size = 2
    fill_value = (-1,)
    input_dict = {"a": a, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D uint8 array
    a = np.array([12, 0, 0, 45, 0], dtype=np.uint8)
    size = 3
    fill_value = (255,)
    input_dict = {"a": a, "size": size, "fill_value": fill_value}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.flatnonzero_4"] = flatnonzero_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.flatnonzero_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.flatnonzero_4'.")


check_valid('jax.numpy.flatnonzero', generated_inputs['jax.numpy.flatnonzero_4'], lib="jax", suffix=4)
