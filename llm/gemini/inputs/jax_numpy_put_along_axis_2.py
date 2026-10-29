
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def put_along_axis_inputs():
    list_of_inputs = []

    # Input 1: 1D array, axis 0, promise_in_bounds
    arr = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    indices = np.array([1, 3], dtype=np.int32)
    values = 99.0
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

    # Input 2: 2D array, axis 0, drop mode
    arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    indices = np.array([[0, 2, 1, 0]], dtype=np.int64)
    values = -5.5
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

    # Input 3: 2D array, axis 1, fill mode
    arr = np.zeros((4, 5), dtype=np.float32)
    indices = np.array([[1], [3], [0], [4]], dtype=np.int32)
    values = 7.0
    axis = 1
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

    # Input 4: 2D array, negative axis -1, promise_in_bounds
    arr = np.ones((2, 3), dtype=np.float32)
    indices = np.array([[2, 0], [1, 2]], dtype=np.int32)
    values = 42.0
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

    # Input 5: 3D array, axis 2, drop mode
    arr = np.random.randn(2, 3, 4).astype(np.float64)
    indices = np.zeros((2, 3, 1), dtype=np.int32)
    values = 0.0
    axis = 2
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

    # Input 6: 3D array, axis 0, fill mode
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    indices = np.ones((1, 3, 4), dtype=np.int64)
    values = -1.0
    axis = 0
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

    # Input 7: 3D array, negative axis -2, promise_in_bounds
    arr = np.linspace(0, 10, 24, dtype=np.float32).reshape(2, 4, 3)
    indices = np.array([[[1, 2, 0]], [[3, 1, 2]]], dtype=np.int32)
    values = 3.14159
    axis = -2
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

    # Input 8: 4D array, axis 1, drop mode
    arr = np.zeros((2, 3, 2, 2), dtype=np.float32)
    indices = np.ones((2, 1, 2, 2), dtype=np.int32)
    values = 8.8
    axis = 1
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

    # Input 9: 1D array with multiple indices to update, fill mode
    arr = np.array([-10.0, -20.0, -30.0, -40.0, -50.0], dtype=np.float64)
    indices = np.array([0, 2, 4], dtype=np.int64)
    values = 100.0
    axis = 0
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

    # Input 10: 2D array, full-slice indices along axis 0, promise_in_bounds
    arr = np.ones((4, 2), dtype=np.float32)
    indices = np.array([[3, 2], [1, 0]], dtype=np.int32)
    values = 2.718
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

    return list_of_inputs


generated_inputs["jax.numpy.put_along_axis_2"] = put_along_axis_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.put_along_axis_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.put_along_axis_2'.")


check_valid('jax.numpy.put_along_axis', generated_inputs['jax.numpy.put_along_axis_2'], lib="jax", suffix=2)
