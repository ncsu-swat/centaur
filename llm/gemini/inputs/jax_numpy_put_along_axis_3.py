
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def put_along_axis_inputs():
    list_of_inputs = []

    # Input 1: 1D array, axis=0, mode='promise_in_bounds'
    arr = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    indices = np.array([1, 3], dtype=np.int64)
    values = 99
    axis = 0
    inplace = False
    mode = "promise_in_bounds"
    list_of_inputs.append(
        copy.deepcopy(
            {
                "arr": arr,
                "indices": indices,
                "values": values,
                "axis": axis,
                "inplace": inplace,
                "mode": mode,
            }
        )
    )

    # Input 2: 2D array, axis=1, mode='clip'
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    indices = np.array([[2], [0]], dtype=np.int64)
    values = -1
    axis = 1
    inplace = False
    mode = "clip"
    list_of_inputs.append(
        copy.deepcopy(
            {
                "arr": arr,
                "indices": indices,
                "values": values,
                "axis": axis,
                "inplace": inplace,
                "mode": mode,
            }
        )
    )

    # Input 3: 2D array, axis=0, negative axis, mode='drop'
    arr = np.zeros((4, 3), dtype=np.int64)
    indices = np.array([[1, 2, 3]], dtype=np.int64)
    values = 7
    axis = -2
    inplace = False
    mode = "drop"
    list_of_inputs.append(
        copy.deepcopy(
            {
                "arr": arr,
                "indices": indices,
                "values": values,
                "axis": axis,
                "inplace": inplace,
                "mode": mode,
            }
        )
    )

    # Input 4: 2D array, float data, axis=-1, mode='fill'
    arr = np.ones((3, 4), dtype=np.float32)
    indices = np.array([[0, 2], [1, 3], [2, 0]], dtype=np.int32)
    values = 0
    axis = -1
    inplace = False
    mode = "fill"
    list_of_inputs.append(
        copy.deepcopy(
            {
                "arr": arr,
                "indices": indices,
                "values": values,
                "axis": axis,
                "inplace": inplace,
                "mode": mode,
            }
        )
    )

    # Input 5: 3D array, axis=1, mode='promise_in_bounds'
    arr = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    indices = np.zeros((2, 1, 4), dtype=np.int64)
    values = 100
    axis = 1
    inplace = False
    mode = "promise_in_bounds"
    list_of_inputs.append(
        copy.deepcopy(
            {
                "arr": arr,
                "indices": indices,
                "values": values,
                "axis": axis,
                "inplace": inplace,
                "mode": mode,
            }
        )
    )

    # Input 6: 3D array, axis=2, mode='clip'
    arr = np.ones((2, 2, 5), dtype=np.int64)
    indices = np.array([[[1, 3], [0, 4]], [[2, 2], [3, 1]]], dtype=np.int32)
    values = -50
    axis = 2
    inplace = False
    mode = "clip"
    list_of_inputs.append(
        copy.deepcopy(
            {
                "arr": arr,
                "indices": indices,
                "values": values,
                "axis": axis,
                "inplace": inplace,
                "mode": mode,
            }
        )
    )

    # Input 7: 3D array, axis=0, mode='drop'
    arr = np.full((3, 2, 2), 5, dtype=np.int32)
    indices = np.array([[[0, 1], [2, 0]]], dtype=np.int64)
    values = 42
    axis = 0
    inplace = False
    mode = "drop"
    list_of_inputs.append(
        copy.deepcopy(
            {
                "arr": arr,
                "indices": indices,
                "values": values,
                "axis": axis,
                "inplace": inplace,
                "mode": mode,
            }
        )
    )

    # Input 8: 4D array, axis=-1, mode='promise_in_bounds'
    arr = np.zeros((2, 2, 2, 3), dtype=np.int32)
    indices = np.ones((2, 2, 2, 1), dtype=np.int64)
    values = 10
    axis = -1
    inplace = False
    mode = "promise_in_bounds"
    list_of_inputs.append(
        copy.deepcopy(
            {
                "arr": arr,
                "indices": indices,
                "values": values,
                "axis": axis,
                "inplace": inplace,
                "mode": mode,
            }
        )
    )

    # Input 9: 1D array, axis=-1, mode='drop'
    arr = np.array([100, 200, 300], dtype=np.int64)
    indices = np.array([2, 1, 0], dtype=np.int64)
    values = 555
    axis = -1
    inplace = False
    mode = "drop"
    list_of_inputs.append(
        copy.deepcopy(
            {
                "arr": arr,
                "indices": indices,
                "values": values,
                "axis": axis,
                "inplace": inplace,
                "mode": mode,
            }
        )
    )

    # Input 10: 2D array, full column replacement, axis=0, mode='clip'
    arr = np.arange(12, dtype=np.int32).reshape(3, 4)
    indices = np.array([[2, 1, 0, 2], [0, 0, 1, 1]], dtype=np.int32)
    values = -999
    axis = 0
    inplace = False
    mode = "clip"
    list_of_inputs.append(
        copy.deepcopy(
            {
                "arr": arr,
                "indices": indices,
                "values": values,
                "axis": axis,
                "inplace": inplace,
                "mode": mode,
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.put_along_axis_3"] = put_along_axis_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.put_along_axis_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.put_along_axis_3'.")


check_valid('jax.numpy.put_along_axis', generated_inputs['jax.numpy.put_along_axis_3'], lib="jax", suffix=3)
