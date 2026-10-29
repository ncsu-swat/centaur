
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def flatnonzero_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array with mixed zeros and non-zeros
    a = np.array([0, 1, 0, 3, 4, 0, 5], dtype=np.int32)
    size = 4
    fill_value = (0,)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 2: 2D float32 array
    a = np.array([[-1.5, 0.0], [2.3, -4.1]], dtype=np.float32)
    size = 5
    fill_value = (-1,)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 3: 1D boolean array
    a = np.array([True, False, True, False, False, True], dtype=bool)
    size = 3
    fill_value = (0,)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 4: 3D int64 array with negative and positive values
    a = np.array(
        [[[0, -2], [3, 0]], [[-5, 6], [0, 0]]],
        dtype=np.int64,
    )
    size = 4
    fill_value = (999,)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 5: 2D float32 array with some zeros
    a = np.array([[0.0, 1.5], [3.0, 0.0]], dtype=np.float32)
    size = 2
    fill_value = (-1,)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 6: 1D float64 array with all zeros
    a = np.zeros((8,), dtype=np.float64)
    size = 3
    fill_value = (0,)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 7: 4D int32 array with sparse non-zeros
    a = np.zeros((2, 2, 2, 2), dtype=np.int32)
    a[0, 1, 0, 1] = 10
    a[1, 0, 1, 0] = -20
    size = 2
    fill_value = (-100,)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 8: 2D int32 array with all non-zero values
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    size = 6
    fill_value = (0,)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 9: 3D float32 array
    a = np.array([[[0.0, 0.0], [5.5, 0.0]]], dtype=np.float32)
    size = 1
    fill_value = (0,)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 10: 1D int64 array
    a = np.array([10, 0, -20, 0, 30, 40], dtype=np.int64)
    size = 10
    fill_value = (0,)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    # Input 11: 2D float64 array with negative values
    a = np.array([[-3.2, 0.0, 4.5], [0.0, -1.1, 0.0]], dtype=np.float64)
    size = 4
    fill_value = (-1,)
    list_of_inputs.append(
        copy.deepcopy({"a": a, "size": size, "fill_value": fill_value})
    )

    return list_of_inputs


generated_inputs["jax.numpy.flatnonzero_4"] = flatnonzero_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.flatnonzero_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.flatnonzero_4'.")


check_valid('jax.numpy.flatnonzero', generated_inputs['jax.numpy.flatnonzero_4'], lib="jax", suffix=4)
