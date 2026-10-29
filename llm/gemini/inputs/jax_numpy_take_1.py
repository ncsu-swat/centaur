
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def take_inputs():
    list_of_inputs = []

    # Input 1: 1D array, 1D indices, mode='fill'
    input_dict = {
        "a": np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32),
        "indices": np.array([0, 2, 4], dtype=np.int32),
        "axis": 0,
        "mode": "fill",
        "unique_indices": True,
        "indices_are_sorted": True,
        "fill_value": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array, 1D indices, axis=1, mode='clip'
    input_dict = {
        "a": np.arange(12, dtype=np.float64).reshape(3, 4),
        "indices": np.array([1, 3], dtype=np.int64),
        "axis": 1,
        "mode": "clip",
        "unique_indices": True,
        "indices_are_sorted": True,
        "fill_value": -1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array, negative axis, duplicate unsorted indices
    input_dict = {
        "a": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        "indices": np.array([2, 0, 2, 1], dtype=np.int32),
        "axis": -2,
        "mode": "fill",
        "unique_indices": False,
        "indices_are_sorted": False,
        "fill_value": np.nan,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, 2D indices matrix
    input_dict = {
        "a": np.linspace(0.0, 1.0, 6, dtype=np.float32).reshape(2, 3),
        "indices": np.array([[0, 1], [1, 2]], dtype=np.int32),
        "axis": 1,
        "mode": "clip",
        "unique_indices": False,
        "indices_are_sorted": False,
        "fill_value": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array, negative indices, mode='fill'
    input_dict = {
        "a": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        "indices": np.array([-1, -3], dtype=np.int32),
        "axis": 1,
        "mode": "fill",
        "unique_indices": False,
        "indices_are_sorted": False,
        "fill_value": -999.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array, out-of-bounds indices with clip
    input_dict = {
        "a": np.ones((4, 4, 4), dtype=np.float64),
        "indices": np.array([0, 5, -5, 2], dtype=np.int64),
        "axis": 2,
        "mode": "clip",
        "unique_indices": False,
        "indices_are_sorted": False,
        "fill_value": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Out-of-bounds indices with fill_value
    input_dict = {
        "a": np.array([100.0, 200.0, 300.0], dtype=np.float32),
        "indices": np.array([0, 10, 1], dtype=np.int32),
        "axis": 0,
        "mode": "fill",
        "unique_indices": False,
        "indices_are_sorted": False,
        "fill_value": 42.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D tensor, axis=3, unique and sorted indices
    input_dict = {
        "a": np.random.randn(2, 3, 2, 5).astype(np.float32),
        "indices": np.array([1, 2, 4], dtype=np.int32),
        "axis": 3,
        "mode": "fill",
        "unique_indices": True,
        "indices_are_sorted": True,
        "fill_value": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array with 0D (scalar-like) index tensor
    input_dict = {
        "a": np.array([7.5, 8.5, 9.5], dtype=np.float64),
        "indices": np.array(1, dtype=np.int32),
        "axis": 0,
        "mode": "clip",
        "unique_indices": True,
        "indices_are_sorted": True,
        "fill_value": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 3D array with negative axis and mode='clip'
    input_dict = {
        "a": np.zeros((3, 5, 2), dtype=np.float32),
        "indices": np.array([0, 1, 2, 3, 4], dtype=np.int64),
        "axis": -2,
        "mode": "clip",
        "unique_indices": True,
        "indices_are_sorted": True,
        "fill_value": 1.0,
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
