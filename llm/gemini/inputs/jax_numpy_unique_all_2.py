
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def unique_all_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array, size larger than unique elements
    x = np.array([3, 1, 2, 3, 4, 1, 5], dtype=np.int32)
    size = 8
    fill_value = 0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 2: 1D integer array with negative values
    x = np.array([-10, -5, 0, 5, -5, 10, -10], dtype=np.int64)
    size = 10
    fill_value = -1
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 3: 2D integer array, size equal to unique elements
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    size = 6
    fill_value = 0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 4: 2D integer array, size smaller than unique elements (truncation)
    x = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.int32)
    size = 3
    fill_value = 0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 5: 3D integer array
    x = np.arange(24, dtype=np.int32).reshape((2, 3, 4)) % 5
    size = 10
    fill_value = -999
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 6: 1D float array with integer fill_value
    x = np.array([1.5, -2.5, 3.0, 1.5, -2.5, 0.0], dtype=np.float32)
    size = 6
    fill_value = -1
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 7: 2D float array (float64)
    x = np.array([[0.1, 0.2], [0.2, 0.3], [0.1, 0.4]], dtype=np.float64)
    size = 5
    fill_value = 0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 8: All elements identical
    x = np.full((3, 3), 7, dtype=np.int32)
    size = 4
    fill_value = 7
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 9: 4D integer array
    x = np.ones((2, 2, 2, 2), dtype=np.int16)
    size = 5
    fill_value = 0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 10: 1D unsigned integer array
    x = np.array([100, 250, 50, 100, 0, 250], dtype=np.uint8)
    size = 6
    fill_value = 255
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    return list_of_inputs


generated_inputs["jax.numpy.unique_all_2"] = unique_all_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unique_all_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unique_all_2'.")


check_valid('jax.numpy.unique_all', generated_inputs['jax.numpy.unique_all_2'], lib="jax", suffix=2)
