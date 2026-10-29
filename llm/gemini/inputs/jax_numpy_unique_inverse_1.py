
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def unique_inverse_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array with duplicates
    x = np.array([3, 4, 1, 3, 1, 2, 5], dtype=np.int32)
    size = 5
    fill_value = np.array(0, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 2: 1D integer array, size larger than number of unique elements
    x = np.array([1, 2, 2, 3], dtype=np.int64)
    size = 6
    fill_value = np.array(-1, dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 3: 2D float32 array
    x = np.array(
        [[1.5, -2.0], [3.0, 1.5], [-2.0, 0.0]], dtype=np.float32
    )
    size = 4
    fill_value = np.array(99.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 4: 1D float64 array with negative values
    x = np.array([-10.5, -5.2, 0.0, 5.2, 10.5, -5.2], dtype=np.float64)
    size = 5
    fill_value = np.array(np.nan, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 5: 3D int16 array
    x = np.arange(24, dtype=np.int16).reshape((2, 3, 4)) % 5
    size = 8
    fill_value = np.array(-99, dtype=np.int16)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 6: Boolean array
    x = np.array([[True, False], [True, True], [False, False]], dtype=bool)
    size = 2
    fill_value = np.array(False, dtype=bool)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 7: 1D uint8 array
    x = np.array([255, 128, 0, 128, 64, 255], dtype=np.uint8)
    size = 6
    fill_value = np.array(0, dtype=np.uint8)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 8: 4D float32 array
    x = np.ones((2, 2, 2, 2), dtype=np.float32)
    size = 3
    fill_value = np.array(-1.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 9: 1D array where size truncates unique values
    x = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    size = 3
    fill_value = np.array(0, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 10: 2D array of all identical elements
    x = np.full((3, 3), 7, dtype=np.int64)
    size = 4
    fill_value = np.array(-1, dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    return list_of_inputs


generated_inputs["jax.numpy.unique_inverse_1"] = unique_inverse_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unique_inverse_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unique_inverse_1'.")


check_valid('jax.numpy.unique_inverse', generated_inputs['jax.numpy.unique_inverse_1'], lib="jax", suffix=1)
