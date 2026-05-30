
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def take_inputs():
    list_of_inputs = []

    # Input 1: Basic 1D array, valid indices, sorted, unique
    a = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    indices = np.array([1, 3], dtype=np.int32)
    input_dict = {
        "a": a,
        "indices": indices,
        "axis": 0,
        "mode": "fill",
        "unique_indices": True,
        "indices_are_sorted": True,
        "fill_value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, taking along axis 1, unsorted, not unique, clip mode
    a = np.random.randn(3, 4).astype(np.float32)
    indices = np.array([2, 0, 2], dtype=np.int32)
    input_dict = {
        "a": a,
        "indices": indices,
        "axis": 1,
        "mode": "clip",
        "unique_indices": False,
        "indices_are_sorted": False,
        "fill_value": -1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Float64 array, taking along axis 0, out of bounds with NaN fill
    a = np.random.randn(5, 5).astype(np.float64)
    indices = np.array([1, 6, -2], dtype=np.int64)
    input_dict = {
        "a": a,
        "indices": indices,
        "axis": 0,
        "mode": "fill",
        "unique_indices": False,
        "indices_are_sorted": False,
        "fill_value": float('nan')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D array, out of bounds with a custom fill value
    a = np.arange(24).reshape(2, 3, 4).astype(np.float32)
    indices = np.array([0, 5], dtype=np.int32)
    input_dict = {
        "a": a,
        "indices": indices,
        "axis": 2,
        "mode": "fill",
        "unique_indices": True,
        "indices_are_sorted": True,
        "fill_value": -99.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Same 3D array, out of bounds with clip mode
    a = np.arange(24).reshape(2, 3, 4).astype(np.float32)
    indices = np.array([0, 5], dtype=np.int32)
    input_dict = {
        "a": a,
        "indices": indices,
        "axis": 2,
        "mode": "clip",
        "unique_indices": True,
        "indices_are_sorted": True,
        "fill_value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D array and 2D indices
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    indices = np.array([[0, 1], [1, 0]], dtype=np.int32)
    input_dict = {
        "a": a,
        "indices": indices,
        "axis": 0,
        "mode": "fill",
        "unique_indices": False,
        "indices_are_sorted": False,
        "fill_value": -1.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array with identical index elements
    a = np.random.randn(10).astype(np.float32)
    indices = np.array([2, 2, 2], dtype=np.int32)
    input_dict = {
        "a": a,
        "indices": indices,
        "axis": 0,
        "mode": "clip",
        "unique_indices": False,
        "indices_are_sorted": True,
        "fill_value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D array, axis 3, sorted, unique indices
    a = np.random.randn(2, 2, 2, 2).astype(np.float32)
    indices = np.array([0, 1], dtype=np.int32)
    input_dict = {
        "a": a,
        "indices": indices,
        "axis": 3,
        "mode": "fill",
        "unique_indices": True,
        "indices_are_sorted": True,
        "fill_value": 1.23
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Single-element array with repeated indexing
    a = np.array([5.0], dtype=np.float32)
    indices = np.array([0, 0, 0], dtype=np.int32)
    input_dict = {
        "a": a,
        "indices": indices,
        "axis": 0,
        "mode": "fill",
        "unique_indices": False,
        "indices_are_sorted": True,
        "fill_value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array, negative axis index
    a = np.random.randn(8, 8).astype(np.float32)
    indices = np.array([1, 3, 5], dtype=np.int32)
    input_dict = {
        "a": a,
        "indices": indices,
        "axis": -1,
        "mode": "clip",
        "unique_indices": True,
        "indices_are_sorted": True,
        "fill_value": 0.0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.take_1"] = take_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.take_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.take_1'.")


check_valid('jax.numpy.take', generated_inputs['jax.numpy.take_1'], lib="jax", suffix=1)
