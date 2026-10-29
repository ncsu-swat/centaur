
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def unique_counts_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array with positive values
    x = np.array([3, 4, 1, 3, 1], dtype=np.int32)
    size = 5
    fill_value = 0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 2: 1D integer array with negative and positive values
    x = np.array([-2, -5, 3, -2, 0, 4, -5], dtype=np.int64)
    size = 10
    fill_value = -1
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 3: 2D integer array
    x = np.array([[1, 2], [2, 3], [3, 4]], dtype=np.int32)
    size = 6
    fill_value = 99
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 4: 3D integer array
    x = np.ones((2, 3, 4), dtype=np.int32) * 7
    size = 3
    fill_value = 0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 5: 1D float array with size equal to unique count
    x = np.array([1.5, -2.3, 1.5, 4.0, -2.3], dtype=np.float32)
    size = 3
    fill_value = 0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 6: 2D float64 array with size larger than unique count
    x = np.array([[0.0, -1.0], [2.5, 0.0]], dtype=np.float64)
    size = 8
    fill_value = -999
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 7: 1D uint8 array
    x = np.array([10, 20, 20, 30, 40, 50, 10], dtype=np.uint8)
    size = 7
    fill_value = 255
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 8: 4D integer array
    x = np.arange(16, dtype=np.int32).reshape(2, 2, 2, 2) % 4
    size = 10
    fill_value = 100
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 9: Size smaller than unique count (truncation)
    x = np.array([5, 1, 9, 3, 7, 2, 8, 4, 6], dtype=np.int32)
    size = 4
    fill_value = 0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 10: 1D boolean array
    x = np.array([True, False, True, True, False], dtype=np.bool_)
    size = 4
    fill_value = 0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 11: All identical elements
    x = np.full((5, 5), -42, dtype=np.int64)
    size = 2
    fill_value = -1
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    return list_of_inputs


generated_inputs["jax.numpy.unique_counts_2"] = unique_counts_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unique_counts_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unique_counts_2'.")


check_valid('jax.numpy.unique_counts', generated_inputs['jax.numpy.unique_counts_2'], lib="jax", suffix=2)
