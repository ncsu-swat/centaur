
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def unique_inverse_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array with duplicates, size matches unique count
    x = np.array([3, 4, 1, 3, 1], dtype=np.int32)
    size = 3
    fill_value = 0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 2: 1D array, size larger than unique count (padding required)
    x = np.array([1, 2, 2, 3], dtype=np.int64)
    size = 5
    fill_value = -1
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 3: 2D array, size smaller than unique count (truncation)
    x = np.array([[10, 20], [30, 40]], dtype=np.int32)
    size = 2
    fill_value = 0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 4: 2D array with negative numbers
    x = np.array([[-5, -1, -5], [0, 2, -1]], dtype=np.int32)
    size = 6
    fill_value = 99
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 5: 3D array of integers
    x = np.arange(12, dtype=np.int64).reshape((2, 2, 3))
    size = 12
    fill_value = 0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 6: Float array with negative and positive values
    x = np.array([-1.5, 2.0, -1.5, 0.0, 3.5], dtype=np.float32)
    size = 6
    fill_value = -999
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 7: Float64 array with duplicates
    x = np.array([[1.0, 1.0], [2.5, 2.5], [3.0, 3.0]], dtype=np.float64)
    size = 4
    fill_value = 100
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 8: All elements identical
    x = np.ones((4, 4), dtype=np.int32) * 7
    size = 3
    fill_value = -1
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 9: 4D tensor with various integers
    x = np.array([[[[1, 2], [1, 2]], [[3, 4], [3, 4]]]], dtype=np.int32)
    size = 8
    fill_value = -10
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 10: 1D array with single element
    x = np.array([42], dtype=np.int64)
    size = 2
    fill_value = 0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 11: 1D array containing zeros and negatives
    x = np.array([0, -0, -10, 0, 10, -10], dtype=np.int32)
    size = 3
    fill_value = 5
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

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
