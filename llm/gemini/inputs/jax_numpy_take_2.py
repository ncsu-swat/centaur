
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def take_inputs():
    list_of_inputs = []

    # Input 1: Basic positive index
    input_dict = {
        "a": np.array([1, 2, 3, 4, 5], dtype=np.int32),
        "indices": 2,
        "axis": 0,
        "mode": "clip",
        "unique_indices": True,
        "indices_are_sorted": True,
        "fill_value": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 3D float array, mode 'fill'
    input_dict = {
        "a": np.random.randn(3, 4, 5).astype(np.float32),
        "indices": 1,
        "axis": 1,
        "mode": "fill",
        "unique_indices": False,
        "indices_are_sorted": False,
        "fill_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative index
    input_dict = {
        "a": np.arange(10, dtype=np.int32).reshape(2, 5),
        "indices": -1,
        "axis": 0,
        "mode": "clip",
        "unique_indices": True,
        "indices_are_sorted": False,
        "fill_value": 999
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D int array with mode 'fill'
    input_dict = {
        "a": np.ones((2, 2, 2), dtype=np.int32),
        "indices": 0,
        "axis": 2,
        "mode": "fill",
        "unique_indices": False,
        "indices_are_sorted": True,
        "fill_value": -99
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Out of bounds index, mode 'clip'
    input_dict = {
        "a": np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32),
        "indices": 3,
        "axis": 0,
        "mode": "clip",
        "unique_indices": True,
        "indices_are_sorted": True,
        "fill_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Out of bounds index, mode 'fill' with negative fill_value
    input_dict = {
        "a": np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32),
        "indices": 3,
        "axis": 0,
        "mode": "fill",
        "unique_indices": False,
        "indices_are_sorted": False,
        "fill_value": -1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Negative axis
    input_dict = {
        "a": np.zeros((5, 5, 5), dtype=np.int32),
        "indices": 4,
        "axis": -1,
        "mode": "clip",
        "unique_indices": True,
        "indices_are_sorted": True,
        "fill_value": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Single-element array
    input_dict = {
        "a": np.array([100], dtype=np.int32),
        "indices": 0,
        "axis": 0,
        "mode": "clip",
        "unique_indices": False,
        "indices_are_sorted": True,
        "fill_value": 10
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative index on axis 1
    input_dict = {
        "a": np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32),
        "indices": -2,
        "axis": 1,
        "mode": "fill",
        "unique_indices": True,
        "indices_are_sorted": False,
        "fill_value": -5
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Larger 3D float array with unique indices and sorted assumptions false
    input_dict = {
        "a": np.arange(24).reshape(2, 3, 4).astype(np.float32),
        "indices": 2,
        "axis": 1,
        "mode": "clip",
        "unique_indices": False,
        "indices_are_sorted": False,
        "fill_value": 42
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.take_2"] = take_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.take_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.take_2'.")


check_valid('jax.numpy.take', generated_inputs['jax.numpy.take_2'], lib="jax", suffix=2)
