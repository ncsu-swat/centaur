
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def take_along_axis_inputs():
    list_of_inputs = []

    # Input 1: 1D array, axis=0, mode="fill"
    arr = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    indices = np.array([4, 0, 2], dtype=np.int32)
    list_of_inputs.append(
        {
            "arr": arr,
            "indices": indices,
            "axis": 0,
            "mode": "fill",
            "fill_value": 0,
        }
    )

    # Input 2: 2D float array, axis=1, mode="clip"
    arr = np.random.randn(3, 4).astype(np.float32)
    indices = np.array([[0, 2], [1, 3], [2, 0]], dtype=np.int64)
    list_of_inputs.append(
        {
            "arr": arr,
            "indices": indices,
            "axis": 1,
            "mode": "clip",
            "fill_value": -1,
        }
    )

    # Input 3: 2D array, negative axis=-1, mode="fill"
    arr = np.arange(12, dtype=np.int64).reshape(3, 4)
    indices = np.array([[3, 2, 1, 0], [0, 1, 2, 3], [1, 1, 1, 1]], dtype=np.int32)
    list_of_inputs.append(
        {
            "arr": arr,
            "indices": indices,
            "axis": -1,
            "mode": "fill",
            "fill_value": 99,
        }
    )

    # Input 4: 2D array, axis=0 with broadcasting
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    indices = np.array([[0, 1, 0]], dtype=np.int32)
    list_of_inputs.append(
        {
            "arr": arr,
            "indices": indices,
            "axis": 0,
            "mode": "fill",
            "fill_value": 0,
        }
    )

    # Input 5: 3D array, axis=2, mode="clip"
    arr = np.random.randn(2, 3, 4).astype(np.float32)
    indices = np.random.randint(0, 4, size=(2, 3, 2)).astype(np.int32)
    list_of_inputs.append(
        {
            "arr": arr,
            "indices": indices,
            "axis": 2,
            "mode": "clip",
            "fill_value": 0,
        }
    )

    # Input 6: 3D array, axis=1 with negative indices, mode="fill"
    arr = np.arange(24, dtype=np.int32).reshape(2, 4, 3)
    indices = np.array(
        [[[0, 1, 2], [3, 2, 1]], [[1, 0, 2], [2, 3, 0]]], dtype=np.int32
    )
    list_of_inputs.append(
        {
            "arr": arr,
            "indices": indices,
            "axis": 1,
            "mode": "fill",
            "fill_value": -999,
        }
    )

    # Input 7: 3D array, axis=-3 (equivalent to axis=0)
    arr = np.random.randn(4, 3, 2).astype(np.float64)
    indices = np.array([[[0, 1]], [[2, 3]]], dtype=np.int64)
    list_of_inputs.append(
        {
            "arr": arr,
            "indices": indices,
            "axis": -3,
            "mode": "clip",
            "fill_value": 1,
        }
    )

    # Input 8: 4D array, axis=3, mode="drop"
    arr = np.random.randn(2, 2, 3, 5).astype(np.float32)
    indices = np.random.randint(0, 5, size=(2, 2, 3, 2)).astype(np.int32)
    list_of_inputs.append(
        {
            "arr": arr,
            "indices": indices,
            "axis": 3,
            "mode": "drop",
            "fill_value": 0,
        }
    )

    # Input 9: 1D array with out-of-bounds indices, mode="fill"
    arr = np.array([100, 200, 300], dtype=np.int32)
    indices = np.array([0, 5, 2, -10], dtype=np.int32)
    list_of_inputs.append(
        {
            "arr": arr,
            "indices": indices,
            "axis": 0,
            "mode": "fill",
            "fill_value": -1,
        }
    )

    # Input 10: 2D array, axis=-2, mode="clip"
    arr = np.array([[5, 6, 7], [8, 9, 10], [11, 12, 13]], dtype=np.int64)
    indices = np.array([[2, 0, 1], [1, 2, 0], [0, 1, 2], [2, 2, 2]], dtype=np.int32)
    list_of_inputs.append(
        {
            "arr": arr,
            "indices": indices,
            "axis": -2,
            "mode": "clip",
            "fill_value": 42,
        }
    )

    # Input 11: 4D array, axis=-1, mode="fill"
    arr = np.zeros((1, 2, 3, 4), dtype=np.int32)
    indices = np.zeros((1, 2, 3, 1), dtype=np.int32)
    list_of_inputs.append(
        {
            "arr": arr,
            "indices": indices,
            "axis": -1,
            "mode": "fill",
            "fill_value": 7,
        }
    )

    return list_of_inputs


generated_inputs["jax.numpy.take_along_axis_2"] = take_along_axis_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.take_along_axis_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.take_along_axis_2'.")


check_valid('jax.numpy.take_along_axis', generated_inputs['jax.numpy.take_along_axis_2'], lib="jax", suffix=2)
