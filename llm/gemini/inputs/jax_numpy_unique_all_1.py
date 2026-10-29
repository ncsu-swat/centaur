
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def unique_all_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array with positive and negative numbers
    x = np.array([3, -1, 4, 1, -1, 5, 9, 2, 6, 5], dtype=np.int32)
    size = 10
    fill_value = np.array(-999, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 2: 2D float32 array with duplicates
    x = np.array([[1.5, -2.0], [3.2, 1.5], [-2.0, 0.0]], dtype=np.float32)
    size = 6
    fill_value = np.array(np.nan, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 3: 1D int64 array where size is smaller than number of unique elements
    x = np.array([100, 200, 300, 400, 500, 600], dtype=np.int64)
    size = 3
    fill_value = np.array(0, dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 4: 3D uint8 array
    x = np.array(
        [[[0, 255], [128, 0]], [[128, 255], [0, 64]]], dtype=np.uint8
    )
    size = 8
    fill_value = np.array(255, dtype=np.uint8)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 5: 2D float64 array with zeros and negative numbers
    x = np.array(
        [[-0.5, 0.0, 1.2], [-0.5, -3.4, 0.0], [1.2, 2.8, -3.4]],
        dtype=np.float64,
    )
    size = 7
    fill_value = np.array(-1.0, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 6: 1D boolean array
    x = np.array([True, False, True, True, False], dtype=np.bool_)
    size = 4
    fill_value = np.array(False, dtype=np.bool_)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 7: 4D int16 array
    x = np.zeros((2, 2, 2, 2), dtype=np.int16)
    x[0, 1, 0, 1] = 5
    x[1, 0, 1, 0] = -3
    size = 5
    fill_value = np.array(-1, dtype=np.int16)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 8: 1D array with all identical elements
    x = np.full((12,), 42, dtype=np.int32)
    size = 3
    fill_value = np.array(0, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 9: 2D int8 array with negative range
    x = np.array([[-128, -64, 0], [64, 127, -128]], dtype=np.int8)
    size = 6
    fill_value = np.array(0, dtype=np.int8)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 10: 1D float32 array with exact match size
    x = np.array([10.0, -5.0, 2.5, -5.0, 10.0], dtype=np.float32)
    size = 3
    fill_value = np.array(-99.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    return list_of_inputs


generated_inputs["jax.numpy.unique_all_1"] = unique_all_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unique_all_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unique_all_1'.")


check_valid('jax.numpy.unique_all', generated_inputs['jax.numpy.unique_all_1'], lib="jax", suffix=1)
