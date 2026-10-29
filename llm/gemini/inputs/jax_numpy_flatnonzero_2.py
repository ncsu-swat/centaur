
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def flatnonzero_inputs():
    list_of_inputs = []

    # Input 1: 1D array with positive and negative integers
    a = np.array([0, -1, 2, 0, -3, 4], dtype=np.int32)
    size = 4
    fill_value = -1
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 2: 2D float array with mixed zeros and non-zeros
    a = np.array([[0.0, 1.5, -2.3], [0.0, 0.0, 4.1]], dtype=np.float32)
    size = 5
    fill_value = 0
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 3: 3D boolean array
    a = np.array(
        [[[True, False], [False, True]], [[False, False], [True, False]]],
        dtype=np.bool_,
    )
    size = 3
    fill_value = 999
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 4: All-zero 1D array
    a = np.zeros((10,), dtype=np.int64)
    size = 5
    fill_value = -1
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 5: All-nonzero 2D array
    a = np.ones((3, 3), dtype=np.float64)
    size = 9
    fill_value = 0
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 6: 4D array with integers
    a = np.arange(16, dtype=np.int32).reshape(2, 2, 2, 2)
    size = 10
    fill_value = -5
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 7: Complex numbers array
    a = np.array([0 + 0j, 1 + 2j, 0 + 0j, -1 - 1j, 3 + 0j], dtype=np.complex64)
    size = 4
    fill_value = 0
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 8: 0D scalar array non-zero
    a = np.array(42, dtype=np.int32)
    size = 2
    fill_value = -1
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 9: 0D scalar array zero
    a = np.array(0.0, dtype=np.float32)
    size = 1
    fill_value = 100
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 10: Size smaller than actual number of non-zero elements (truncation)
    a = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
    size = 3
    fill_value = 0
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 11: 2D array with large fill_value
    a = np.array([[0, 0], [1, 0]], dtype=np.int8)
    size = 3
    fill_value = 10000
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    return list_of_inputs


generated_inputs["jax.numpy.flatnonzero_2"] = flatnonzero_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.flatnonzero_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.flatnonzero_2'.")


check_valid('jax.numpy.flatnonzero', generated_inputs['jax.numpy.flatnonzero_2'], lib="jax", suffix=2)
