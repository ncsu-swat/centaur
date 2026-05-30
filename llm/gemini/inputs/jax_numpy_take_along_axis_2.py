
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_take_along_axis_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array with positive indices
    arr = np.array([10, 20, 30], dtype=np.int32)
    indices = np.array([2, 0, 1], dtype=np.int32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "axis": 0,
        "mode": "fill",
        "fill_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, axis=1, clip mode
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    indices = np.array([[0, 2], [1, 1]], dtype=np.int32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "axis": 1,
        "mode": "clip",
        "fill_value": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, axis=0, fill mode with custom fill_value
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    indices = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "axis": 0,
        "mode": "fill",
        "fill_value": 9
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Out-of-bounds indices with clip mode on 1D
    arr = np.array([10, 20, 30], dtype=np.int32)
    indices = np.array([4, -1, 3], dtype=np.int32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "axis": 0,
        "mode": "clip",
        "fill_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array with axis=2
    arr = np.random.randn(2, 2, 3).astype(np.float32)
    indices = np.random.randint(0, 3, size=(2, 2, 2)).astype(np.int32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "axis": 2,
        "mode": "fill",
        "fill_value": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array with negative axis
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    indices = np.array([[0, 1], [2, 0]], dtype=np.int32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "axis": -1,
        "mode": "fill",
        "fill_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Float64 2D array
    arr = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    indices = np.array([[1, 0], [0, 1]], dtype=np.int32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "axis": 1,
        "mode": "fill",
        "fill_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D arrays with axis=3
    arr = np.random.randint(0, 10, size=(2, 2, 2, 2)).astype(np.int32)
    indices = np.random.randint(0, 2, size=(2, 2, 2, 2)).astype(np.int32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "axis": 3,
        "mode": "clip",
        "fill_value": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Out of bounds indices with fill mode and a negative fill_value
    arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
    indices = np.array([[0, 5], [1, 2]], dtype=np.int32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "axis": 1,
        "mode": "fill",
        "fill_value": -999
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D with negative indices to take elements from the end
    arr = np.array([10, 20, 30, 40], dtype=np.int32)
    indices = np.array([-1, -2, -3], dtype=np.int32)
    input_dict = {
        "arr": arr,
        "indices": indices,
        "axis": 0,
        "mode": "fill",
        "fill_value": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.take_along_axis_2"] = jax_numpy_take_along_axis_inputs()

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
