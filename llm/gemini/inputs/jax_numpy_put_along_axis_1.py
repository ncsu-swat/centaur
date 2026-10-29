
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def put_along_axis_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis 0, promise_in_bounds
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    indices = np.array([0, 3], dtype=np.int32)
    values = np.array([-10.0, -40.0], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "arr": arr,
                "indices": indices,
                "values": values,
                "axis": 0,
                "inplace": False,
                "mode": "promise_in_bounds",
            }
        )
    )

    # Input 2: 2D int32 array, axis 1, clip mode
    arr = np.arange(12, dtype=np.int32).reshape(3, 4)
    indices = np.array([[1, 2], [0, 3], [2, 1]], dtype=np.int32)
    values = np.array([[99, 98], [97, 96], [95, 94]], dtype=np.int32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "arr": arr,
                "indices": indices,
                "values": values,
                "axis": 1,
                "inplace": False,
                "mode": "clip",
            }
        )
    )

    # Input 3: 2D float64 array, axis 0, drop mode
    arr = np.ones((4, 3), dtype=np.float64)
    indices = np.array([[0, 2, 1]], dtype=np.int64)
    values = np.array([[5.0, 6.0, 7.0]], dtype=np.float64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "arr": arr,
                "indices": indices,
                "values": values,
                "axis": 0,
                "inplace": False,
                "mode": "drop",
            }
        )
    )

    # Input 4: 3D int64 array, axis 2, fill mode
    arr = np.zeros((2, 3, 4), dtype=np.int64)
    indices = np.array(
        [[[1], [2], [0]], [[3], [1], [2]]],
        dtype=np.int64,
    )
    values = np.array(
        [[[42], [43], [44]], [[45], [46], [47]]],
        dtype=np.int64,
    )
    list_of_inputs.append(
        copy.deepcopy(
            {
                "arr": arr,
                "indices": indices,
                "values": values,
                "axis": 2,
                "inplace": False,
                "mode": "fill",
            }
        )
    )

    # Input 5: 3D float32 array, axis -1 (negative axis), promise_in_bounds
    arr = np.random.randn(2, 4, 3).astype(np.float32)
    indices = np.random.randint(0, 3, size=(2, 4, 2), dtype=np.int32)
    values = np.random.randn(2, 4, 2).astype(np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "arr": arr,
                "indices": indices,
                "values": values,
                "axis": -1,
                "inplace": False,
                "mode": "promise_in_bounds",
            }
        )
    )

    # Input 6: 4D float32 array, axis 1, clip mode
    arr = np.zeros((2, 5, 2, 2), dtype=np.float32)
    indices = np.ones((2, 1, 2, 2), dtype=np.int32)
    values = np.full((2, 1, 2, 2), 3.14, dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "arr": arr,
                "indices": indices,
                "values": values,
                "axis": 1,
                "inplace": False,
                "mode": "clip",
            }
        )
    )

    # Input 7: 2D int16 array with negative indices, axis 1, drop mode
    arr = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int16)
    indices = np.array([[-1], [-2]], dtype=np.int32)
    values = np.array([[99], [88]], dtype=np.int16)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "arr": arr,
                "indices": indices,
                "values": values,
                "axis": 1,
                "inplace": False,
                "mode": "drop",
            }
        )
    )

    # Input 8: 1D bool array, axis 0, fill mode
    arr = np.array([True, False, True, False], dtype=np.bool_)
    indices = np.array([1, 3], dtype=np.int64)
    values = np.array([True, True], dtype=np.bool_)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "arr": arr,
                "indices": indices,
                "values": values,
                "axis": 0,
                "inplace": False,
                "mode": "fill",
            }
        )
    )

    # Input 9: 3D complex64 array, axis 0, promise_in_bounds
    arr = np.zeros((3, 2, 2), dtype=np.complex64)
    indices = np.array([[[1, 2], [0, 1]]], dtype=np.int32)
    values = np.array([[[1.0 + 2.0j, 3.0 + 4.0j], [5.0 + 6.0j, 7.0 + 8.0j]]], dtype=np.complex64)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "arr": arr,
                "indices": indices,
                "values": values,
                "axis": 0,
                "inplace": False,
                "mode": "promise_in_bounds",
            }
        )
    )

    # Input 10: 2D float32 array, axis -2, clip mode
    arr = np.ones((5, 3), dtype=np.float32)
    indices = np.array([[0, 2, 4], [1, 3, 2]], dtype=np.int32)
    values = np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], dtype=np.float32)
    list_of_inputs.append(
        copy.deepcopy(
            {
                "arr": arr,
                "indices": indices,
                "values": values,
                "axis": -2,
                "inplace": False,
                "mode": "clip",
            }
        )
    )

    return list_of_inputs


generated_inputs["jax.numpy.put_along_axis_1"] = put_along_axis_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.put_along_axis_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.put_along_axis_1'.")


check_valid('jax.numpy.put_along_axis', generated_inputs['jax.numpy.put_along_axis_1'], lib="jax", suffix=1)
