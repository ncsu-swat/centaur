
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def unique_values_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array
    x = np.array([3, 1, 2, 3, 2, 4, 1], dtype=np.int32)
    size = 6
    fill_value = np.array(-1, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 2: 2D float32 array with negative numbers
    x = np.array([[-1.5, 2.0], [0.0, -1.5]], dtype=np.float32)
    size = 5
    fill_value = np.array(-999.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 3: 1D int64 array, size smaller than number of unique elements
    x = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    size = 3
    fill_value = np.array(0, dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 4: 3D float64 array
    x = np.array(
        [[[1.0, 2.0], [3.0, 1.0]], [[2.0, 4.0], [5.0, 6.0]]], dtype=np.float64
    )
    size = 8
    fill_value = np.array(0.0, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 5: 1D bool array
    x = np.array([True, False, True, True, False], dtype=bool)
    size = 4
    fill_value = np.array(False, dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 6: 2D int16 array with duplicate and negative values
    x = np.array([[-5, -10], [-5, 0]], dtype=np.int16)
    size = 4
    fill_value = np.array(-1, dtype=np.int16)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 7: 1D float32 array with NaN and inf values
    x = np.array([np.nan, 1.0, np.inf, np.nan, -np.inf], dtype=np.float32)
    size = 6
    fill_value = np.array(0.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 8: 4D uint8 array
    x = np.arange(16, dtype=np.uint8).reshape((2, 2, 2, 2))
    size = 20
    fill_value = np.array(255, dtype=np.uint8)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 9: 1D array, size=1 (truncating to first unique element)
    x = np.array([5, 4, 3, 2, 1], dtype=np.int32)
    size = 1
    fill_value = np.array(0, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 10: 2D float64 array with identical values
    x = np.ones((4, 4), dtype=np.float64)
    size = 3
    fill_value = np.array(-1.0, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    return list_of_inputs


generated_inputs["jax.numpy.unique_values_1"] = unique_values_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unique_values_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unique_values_1'.")


check_valid('jax.numpy.unique_values', generated_inputs['jax.numpy.unique_values_1'], lib="jax", suffix=1)
