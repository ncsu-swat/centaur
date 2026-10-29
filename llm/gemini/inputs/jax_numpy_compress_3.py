
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def compress_inputs():
    list_of_inputs = []

    # Input 1: 1D array, int32
    condition = np.array([True, False, True, True, False])
    a = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    axis = 0
    size = 3
    fill_value = np.array(0, dtype=np.int32)
    list_of_inputs.append(
        {
            "condition": condition,
            "a": a,
            "axis": axis,
            "size": size,
            "fill_value": fill_value,
        }
    )

    # Input 2: 2D array, compress along axis 0 (rows), float32
    condition = np.array([True, False, True])
    a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    axis = 0
    size = 2
    fill_value = np.array(-1.0, dtype=np.float32)
    list_of_inputs.append(
        {
            "condition": condition,
            "a": a,
            "axis": axis,
            "size": size,
            "fill_value": fill_value,
        }
    )

    # Input 3: 2D array, compress along axis 1 (columns), negative axis
    condition = np.array([False, True, True, False])
    a = np.arange(12, dtype=np.int64).reshape(3, 4)
    axis = -1
    size = 2
    fill_value = np.array(999, dtype=np.int64)
    list_of_inputs.append(
        {
            "condition": condition,
            "a": a,
            "axis": axis,
            "size": size,
            "fill_value": fill_value,
        }
    )

    # Input 4: 3D array, compress along middle axis
    condition = np.array([True, True, False])
    a = np.random.randn(2, 3, 4).astype(np.float64)
    axis = 1
    size = 2
    fill_value = np.array(0.0, dtype=np.float64)
    list_of_inputs.append(
        {
            "condition": condition,
            "a": a,
            "axis": axis,
            "size": size,
            "fill_value": fill_value,
        }
    )

    # Input 5: Condition shorter than dimension length
    condition = np.array([True, False])
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    axis = 0
    size = 1
    fill_value = np.array(0, dtype=np.int32)
    list_of_inputs.append(
        {
            "condition": condition,
            "a": a,
            "axis": axis,
            "size": size,
            "fill_value": fill_value,
        }
    )

    # Input 6: Condition longer than dimension length
    condition = np.array([True, False, True, True, True])
    a = np.array([[10, 20], [30, 40]], dtype=np.int16)
    axis = 1
    size = 2
    fill_value = np.array(-1, dtype=np.int16)
    list_of_inputs.append(
        {
            "condition": condition,
            "a": a,
            "axis": axis,
            "size": size,
            "fill_value": fill_value,
        }
    )

    # Input 7: All condition elements False, padding with fill_value
    condition = np.array([False, False, False])
    a = np.array([100, 200, 300], dtype=np.int32)
    axis = 0
    size = 2
    fill_value = np.array(-99, dtype=np.int32)
    list_of_inputs.append(
        {
            "condition": condition,
            "a": a,
            "axis": axis,
            "size": size,
            "fill_value": fill_value,
        }
    )

    # Input 8: 4D array, axis=2
    condition = np.array([1, 0, 1, 0, 1], dtype=np.int32)
    a = np.ones((2, 2, 5, 3), dtype=np.float32)
    axis = 2
    size = 3
    fill_value = np.array(np.nan, dtype=np.float32)
    list_of_inputs.append(
        {
            "condition": condition,
            "a": a,
            "axis": axis,
            "size": size,
            "fill_value": fill_value,
        }
    )

    # Input 9: Complex numbers array
    condition = np.array([True, False, True, False])
    a = np.array(
        [1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j, 7.0 + 8.0j], dtype=np.complex64
    )
    axis = 0
    size = 2
    fill_value = np.array(0.0 + 0.0j, dtype=np.complex64)
    list_of_inputs.append(
        {
            "condition": condition,
            "a": a,
            "axis": axis,
            "size": size,
            "fill_value": fill_value,
        }
    )

    # Input 10: Boolean array compressed
    condition = np.array([True, False, True, False])
    a = np.array([[True, False, True, False], [False, True, False, True]])
    axis = 1
    size = 2
    fill_value = np.array(False, dtype=bool)
    list_of_inputs.append(
        {
            "condition": condition,
            "a": a,
            "axis": axis,
            "size": size,
            "fill_value": fill_value,
        }
    )

    return list_of_inputs


generated_inputs["jax.numpy.compress_3"] = compress_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.compress_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.compress_3'.")


check_valid('jax.numpy.compress', generated_inputs['jax.numpy.compress_3'], lib="jax", suffix=3)
