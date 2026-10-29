
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def unique_counts_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, size > number of unique elements
    x = np.array([1.0, 2.0, 2.0, 3.0, 1.0], dtype=np.float32)
    size = 5
    fill_value = -1.0
    list_of_inputs.append(
        {"x": copy.deepcopy(x), "size": size, "fill_value": fill_value}
    )

    # Input 2: 1D float64 array with negative values, size == number of unique elements
    x = np.array([-10.5, -5.0, 0.0, -5.0, 12.3], dtype=np.float64)
    size = 4
    fill_value = 0.0
    list_of_inputs.append(
        {"x": copy.deepcopy(x), "size": size, "fill_value": fill_value}
    )

    # Input 3: 2D array, size < number of unique elements (truncation)
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    size = 3
    fill_value = 999.0
    list_of_inputs.append(
        {"x": copy.deepcopy(x), "size": size, "fill_value": fill_value}
    )

    # Input 4: 3D float32 array
    x = np.ones((2, 3, 4), dtype=np.float32) * 4.5
    size = 4
    fill_value = -99.5
    list_of_inputs.append(
        {"x": copy.deepcopy(x), "size": size, "fill_value": fill_value}
    )

    # Input 5: 1D array with NaN values
    x = np.array([np.nan, 1.0, np.nan, 2.5, 1.0], dtype=np.float32)
    size = 6
    fill_value = -1.0
    list_of_inputs.append(
        {"x": copy.deepcopy(x), "size": size, "fill_value": fill_value}
    )

    # Input 6: Empty 1D array
    x = np.array([], dtype=np.float32)
    size = 3
    fill_value = 0.0
    list_of_inputs.append(
        {"x": copy.deepcopy(x), "size": size, "fill_value": fill_value}
    )

    # Input 7: 4D float64 array
    x = np.arange(16, dtype=np.float64).reshape((2, 2, 2, 2))
    size = 20
    fill_value = np.nan
    list_of_inputs.append(
        {"x": copy.deepcopy(x), "size": size, "fill_value": fill_value}
    )

    # Input 8: 1D array with identical elements
    x = np.full((10,), 7.7, dtype=np.float32)
    size = 1
    fill_value = 0.0
    list_of_inputs.append(
        {"x": copy.deepcopy(x), "size": size, "fill_value": fill_value}
    )

    # Input 9: 2D array with negative floats and zero
    x = np.array([[-3.0, 0.0, -3.0], [0.0, -1.5, -1.5]], dtype=np.float64)
    size = 5
    fill_value = 100.0
    list_of_inputs.append(
        {"x": copy.deepcopy(x), "size": size, "fill_value": fill_value}
    )

    # Input 10: 1D float16 array
    x = np.array([0.5, -0.5, 0.5, 1.5, -1.5], dtype=np.float16)
    size = 6
    fill_value = -99.0
    list_of_inputs.append(
        {"x": copy.deepcopy(x), "size": size, "fill_value": fill_value}
    )

    # Input 11: 1D array with size=0
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    size = 0
    fill_value = 0.0
    list_of_inputs.append(
        {"x": copy.deepcopy(x), "size": size, "fill_value": fill_value}
    )

    return list_of_inputs


generated_inputs["jax.numpy.unique_counts_3"] = unique_counts_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unique_counts_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unique_counts_3'.")


check_valid('jax.numpy.unique_counts', generated_inputs['jax.numpy.unique_counts_3'], lib="jax", suffix=3)
