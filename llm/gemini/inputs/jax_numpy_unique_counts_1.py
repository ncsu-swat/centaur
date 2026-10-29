
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def unique_counts_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array, size larger than unique elements
    x = np.array([3, 1, 2, 3, 1, 4], dtype=np.int32)
    size = 6
    fill_value = np.array(0, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 2: 1D float32 array, size exactly equal to number of unique elements
    x = np.array([-1.5, 0.0, 2.5, -1.5, 2.5], dtype=np.float32)
    size = 3
    fill_value = np.array(-999.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 3: 2D int64 array with negative numbers
    x = np.array([[-5, 10, -5], [20, 10, 30]], dtype=np.int64)
    size = 5
    fill_value = np.array(-1, dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 4: 1D float64 array containing NaNs
    x = np.array([np.nan, 1.0, 2.0, np.nan, 1.0], dtype=np.float64)
    size = 4
    fill_value = np.array(0.0, dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 5: 3D uint8 array
    x = np.array(
        [[[1, 2], [3, 4]], [[1, 2], [5, 6]]],
        dtype=np.uint8,
    )
    size = 10
    fill_value = np.array(255, dtype=np.uint8)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 6: 1D boolean array
    x = np.array([True, False, True, True, False], dtype=np.bool_)
    size = 2
    fill_value = np.array(False, dtype=np.bool_)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 7: 1D int16 array with size smaller than unique elements
    x = np.array([10, 20, 30, 40, 50], dtype=np.int16)
    size = 3
    fill_value = np.array(0, dtype=np.int16)
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

    # Input 9: 1D float32 array with all identical elements
    x = np.array([7.0, 7.0, 7.0, 7.0], dtype=np.float32)
    size = 4
    fill_value = np.array(0.0, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    # Input 10: 2D int32 array with a larger padding size
    x = np.array([[100, -200], [300, -200]], dtype=np.int32)
    size = 8
    fill_value = np.array(999, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"x": x, "size": size, "fill_value": fill_value})
    )

    return list_of_inputs


generated_inputs["jax.numpy.unique_counts_1"] = unique_counts_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.unique_counts_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.unique_counts_1'.")


check_valid('jax.numpy.unique_counts', generated_inputs['jax.numpy.unique_counts_1'], lib="jax", suffix=1)
