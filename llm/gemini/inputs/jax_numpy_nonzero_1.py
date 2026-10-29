
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def nonzero_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array with positive and zero values
    a = np.array([0, 5, 0, 6, 0, 7], dtype=np.int32)
    size = 5
    fill_value = np.array(-1, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 2: 2D float32 array with negative and positive values
    a = np.array(
        [[0.0, -1.5, 0.0], [2.3, 0.0, -4.1], [0.0, 0.0, 5.0]], dtype=np.float32
    )
    size = 6
    fill_value = np.array(0, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 3: 1D boolean array (acting as a mask)
    a = np.array([True, False, True, True, False], dtype=np.bool_)
    size = 4
    fill_value = np.array(99, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 4: 3D int64 array
    a = np.zeros((2, 3, 4), dtype=np.int64)
    a[0, 1, 2] = 10
    a[1, 0, 3] = -20
    a[1, 2, 1] = 30
    size = 5
    fill_value = np.array(-1, dtype=np.int64)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 5: 2D complex array
    a = np.array(
        [[0 + 0j, 1 + 2j], [-1 - 1j, 0 + 0j], [0 + 3j, 0 + 0j]],
        dtype=np.complex64,
    )
    size = 3
    fill_value = np.array(0, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 6: 4D float64 array
    a = np.zeros((2, 2, 2, 2), dtype=np.float64)
    a[0, 1, 0, 1] = 3.14
    a[1, 0, 1, 0] = -2.71
    size = 4
    fill_value = np.array(-1, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 7: 1D array where size causes truncation (size < number of nonzeros)
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
    size = 3
    fill_value = np.array(0, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 8: 2D array of all zeros
    a = np.zeros((4, 4), dtype=np.int16)
    size = 4
    fill_value = np.array(-1, dtype=np.int16)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 9: 1D uint8 array with all non-zero elements
    a = np.array([255, 128, 64, 32], dtype=np.uint8)
    size = 6
    fill_value = np.array(0, dtype=np.uint8)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 10: 2D float32 array with exact size matching non-zero count
    a = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    size = 2
    fill_value = np.array(-1, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 11: 3D int32 array with negative elements and padding
    a = np.array(
        [[[-1, 0], [0, -2]], [[0, 0], [-3, 0]]],
        dtype=np.int32,
    )
    size = 6
    fill_value = np.array(100, dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    return list_of_inputs


generated_inputs["jax.numpy.nonzero_1"] = nonzero_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.nonzero_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.nonzero_1'.")


check_valid('jax.numpy.nonzero', generated_inputs['jax.numpy.nonzero_1'], lib="jax", suffix=1)
