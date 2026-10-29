
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def unique_values_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array with duplicates
    x = np.array([3.0, 1.0, 2.0, 1.0, 3.0, 4.0], dtype=np.float32)
    size = 6
    fill_value = 0.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 2: 1D int32 array with negative and positive values
    x = np.array([-5, 10, -5, 0, 2, 10, -1], dtype=np.int32)
    size = 10
    fill_value = -999.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 3: 2D float64 array
    x = np.array([[1.5, 2.5], [2.5, 3.5], [1.5, 4.5]], dtype=np.float64)
    size = 4
    fill_value = -1.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 4: 2D float32 array where size is smaller than unique count
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    size = 3
    fill_value = 0.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 5: 3D int64 array
    x = np.arange(24, dtype=np.int64).reshape((2, 3, 4)) % 5
    size = 8
    fill_value = -1.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 6: 1D float32 array with all identical elements
    x = np.full((10,), 7.5, dtype=np.float32)
    size = 5
    fill_value = 0.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 7: 1D array of single element
    x = np.array([42.0], dtype=np.float32)
    size = 3
    fill_value = 1.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 8: 4D float32 tensor
    x = np.ones((2, 2, 2, 2), dtype=np.float32)
    x[0, 1, 0, 1] = -3.0
    x[1, 0, 1, 0] = 5.0
    size = 5
    fill_value = -10.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 9: 1D float64 array with strictly negative values
    x = np.array([-10.0, -20.0, -10.0, -30.0, -20.0], dtype=np.float64)
    size = 5
    fill_value = 99.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 10: Empty 1D float32 array
    x = np.array([], dtype=np.float32)
    size = 4
    fill_value = 0.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 11: 2D int16 array with large padding size
    x = np.array([[3, 3, 3], [3, 3, 3]], dtype=np.int16)
    size = 6
    fill_value = 2.0
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    return list_of_inputs


generated_inputs["jax.numpy.unique_values_3"] = unique_values_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unique_values_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unique_values_3'.")


check_valid('jax.numpy.unique_values', generated_inputs['jax.numpy.unique_values_3'], lib="jax", suffix=3)
