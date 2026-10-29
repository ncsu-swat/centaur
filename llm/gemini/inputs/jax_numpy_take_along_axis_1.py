
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def take_along_axis_inputs():
    list_of_inputs = []

    # Input 1: 2D float array, axis 0, fill mode
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    indices = np.array([[1, 0, 0]], dtype=np.int32)
    axis = 0
    mode = "fill"
    fill_value = 0.0
    list_of_inputs.append(
        {
            "arr": arr,
            "indices": indices,
            "axis": axis,
            "mode": mode,
            "fill_value": fill_value,
        }
    )

    # Input 2: 2D float array, axis 1, clip mode
    arr = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float64)
    indices = np.array([[2, 0], [1, 2]], dtype=np.int64)
    axis = 1
    mode = "clip"
    fill_value = -1.0
    list_of_inputs.append(
        {
            "arr": arr,
            "indices": indices,
            "axis": axis,
            "mode": mode,
            "fill_value": fill_value,
        }
    )

    # Input 3: 1D array, axis 0, fill mode with out-of-bounds index
    arr = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    indices = np.array([3, 0, 10, -1], dtype=np.int32)
    axis = 0
    mode = "fill"
    fill_value = float(np.nan)
    list_of_inputs.append(
        {
            "arr": arr,
            "indices": indices,
            "axis": axis,
            "mode": mode,
            "fill_value": fill_value,
        }
    )

    # Input 4: 3D array, negative axis (-1), clip mode
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    indices = np.zeros((2, 3, 2), dtype=np.int32)
    axis = -1
    mode = "clip"
    fill_value = 0.0
    list_of_inputs.append(
        {
            "arr": arr,
            "indices": indices,
            "axis": axis,
            "mode": mode,
            "fill_value": fill_value,
        }
    )

    # Input 5: 3D array, axis 1, fill mode with custom fill_value
    arr = np.random.randn(3, 4, 5).astype(np.float64)
    indices = np.random.randint(0, 4, size=(3, 2, 5)).astype(np.int64)
    axis = 1
    mode = "fill"
    fill_value = -999.0
    list_of_inputs.append(
        {
            "arr": arr,
            "indices": indices,
            "axis": axis,
            "mode": mode,
            "fill_value": fill_value,
        }
    )

    # Input 6: 4D array, axis 2, fill mode
    arr = np.ones((2, 2, 4, 3), dtype=np.float32)
    indices = np.zeros((2, 2, 3, 3), dtype=np.int32)
    axis = 2
    mode = "fill"
    fill_value = 1.0
    list_of_inputs.append(
        {
            "arr": arr,
            "indices": indices,
            "axis": axis,
            "mode": mode,
            "fill_value": fill_value,
        }
    )

    # Input 7: 2D array, axis -2, clip mode
    arr = np.array([[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0]], dtype=np.float32)
    indices = np.array([[2, 0], [1, 1]], dtype=np.int32)
    axis = -2
    mode = "clip"
    fill_value = 0.5
    list_of_inputs.append(
        {
            "arr": arr,
            "indices": indices,
            "axis": axis,
            "mode": mode,
            "fill_value": fill_value,
        }
    )

    # Input 8: Broadcasting along non-axis dimensions
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    indices = np.array([[0, 1, 2]], dtype=np.int32)
    axis = 1
    mode = "fill"
    fill_value = 0.0
    list_of_inputs.append(
        {
            "arr": arr,
            "indices": indices,
            "axis": axis,
            "mode": mode,
            "fill_value": fill_value,
        }
    )

    # Input 9: 3D array, axis 0, clip mode with negative indices
    arr = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    indices = np.array(
        [[[-1, 0], [1, -2]], [[0, 1], [-1, 0]]], dtype=np.int64
    )
    axis = 0
    mode = "clip"
    fill_value = 42.0
    list_of_inputs.append(
        {
            "arr": arr,
            "indices": indices,
            "axis": axis,
            "mode": mode,
            "fill_value": fill_value,
        }
    )

    # Input 10: 1D array, axis -1, clip mode
    arr = np.linspace(-5.0, 5.0, 10, dtype=np.float64)
    indices = np.array([0, 9, 4, 2], dtype=np.int64)
    axis = -1
    mode = "clip"
    fill_value = 0.0
    list_of_inputs.append(
        {
            "arr": arr,
            "indices": indices,
            "axis": axis,
            "mode": mode,
            "fill_value": fill_value,
        }
    )

    return list_of_inputs


generated_inputs["jax.numpy.take_along_axis_1"] = take_along_axis_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.take_along_axis_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.take_along_axis_1'.")


check_valid('jax.numpy.take_along_axis', generated_inputs['jax.numpy.take_along_axis_1'], lib="jax", suffix=1)
