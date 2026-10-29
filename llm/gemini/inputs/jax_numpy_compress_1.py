
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def compress_inputs():
    list_of_inputs = []

    # Input 1: 1D array, standard boolean mask
    condition = np.array([True, False, True, True], dtype=bool)
    a = np.array([10, 20, 30, 40], dtype=np.int32)
    axis = 0
    size = 3
    fill_value = 0
    list_of_inputs.append(
        copy.deepcopy(
            {
                "condition": condition,
                "a": a,
                "axis": axis,
                "size": size,
                "fill_value": fill_value,
            }
        )
    )

    # Input 2: 2D array, compressing along axis 0
    condition = np.array([True, False, True], dtype=bool)
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    axis = 0
    size = 2
    fill_value = -1
    list_of_inputs.append(
        copy.deepcopy(
            {
                "condition": condition,
                "a": a,
                "axis": axis,
                "size": size,
                "fill_value": fill_value,
            }
        )
    )

    # Input 3: 2D array, compressing along axis 1 with fill_value
    condition = np.array([False, True, True, False], dtype=bool)
    a = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.int64
    )
    axis = 1
    size = 4
    fill_value = 99
    list_of_inputs.append(
        copy.deepcopy(
            {
                "condition": condition,
                "a": a,
                "axis": axis,
                "size": size,
                "fill_value": fill_value,
            }
        )
    )

    # Input 4: Negative axis (-1) on a 3D array
    condition = np.array([1, 0, 1, 1, 0], dtype=np.int32)
    a = np.arange(30, dtype=np.float64).reshape(2, 3, 5)
    axis = -1
    size = 3
    fill_value = 0
    list_of_inputs.append(
        copy.deepcopy(
            {
                "condition": condition,
                "a": a,
                "axis": axis,
                "size": size,
                "fill_value": fill_value,
            }
        )
    )

    # Input 5: Condition shorter than array axis size
    condition = np.array([True, False], dtype=bool)
    a = np.arange(24, dtype=np.int32).reshape(4, 6)
    axis = 0
    size = 2
    fill_value = -999
    list_of_inputs.append(
        copy.deepcopy(
            {
                "condition": condition,
                "a": a,
                "axis": axis,
                "size": size,
                "fill_value": fill_value,
            }
        )
    )

    # Input 6: Condition longer than array axis size
    condition = np.array([True, True, True, True, False], dtype=bool)
    a = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    axis = 1
    size = 2
    fill_value = 1
    list_of_inputs.append(
        copy.deepcopy(
            {
                "condition": condition,
                "a": a,
                "axis": axis,
                "size": size,
                "fill_value": fill_value,
            }
        )
    )

    # Input 7: 4D array, compressing along axis 2
    condition = np.array([True, False, True, False], dtype=bool)
    a = np.ones((2, 3, 4, 5), dtype=np.int16)
    axis = 2
    size = 2
    fill_value = 0
    list_of_inputs.append(
        copy.deepcopy(
            {
                "condition": condition,
                "a": a,
                "axis": axis,
                "size": size,
                "fill_value": fill_value,
            }
        )
    )

    # Input 8: All false condition with size matching axis
    condition = np.array([False, False, False], dtype=bool)
    a = np.array([100, 200, 300], dtype=np.int32)
    axis = 0
    size = 3
    fill_value = 7
    list_of_inputs.append(
        copy.deepcopy(
            {
                "condition": condition,
                "a": a,
                "axis": axis,
                "size": size,
                "fill_value": fill_value,
            }
        )
    )

    # Input 9: All true condition, size equal to condition length
    condition = np.array([True, True, True], dtype=bool)
    a = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    axis = -2
    size = 3
    fill_value = 0
    list_of_inputs.append(
        copy.deepcopy(
            {
                "condition": condition,
                "a": a,
                "axis": axis,
                "size": size,
                "fill_value": fill_value,
            }
        )
    )

    # Input 10: Size smaller than axis size with padding required
    condition = np.array([True, False, False, False, False], dtype=bool)
    a = np.arange(25, dtype=np.float32).reshape(5, 5)
    axis = 0
    size = 3
    fill_value = -5
    list_of_inputs.append(
        copy.deepcopy(
            {
                "condition": condition,
                "a": a,
                "axis": axis,
                "size": size,
                "fill_value": fill_value,
            }
        )
    )

    # Input 11: 3D array with axis 1
    condition = np.array([True, True, False, True], dtype=bool)
    a = np.arange(24, dtype=np.int32).reshape(2, 4, 3)
    axis = 1
    size = 4
    fill_value = 42
    list_of_inputs.append(
        copy.deepcopy(
            {
                "condition": condition,
                "a": a,
                "axis": axis,
                "size": size,
                "fill_value": fill_value,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.compress_1"] = compress_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.compress_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.compress_1'.")


check_valid('jax.numpy.compress', generated_inputs['jax.numpy.compress_1'], lib="jax", suffix=1)
