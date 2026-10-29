
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def take_inputs():
    list_of_inputs = []

    # Input 1: 1D array, sorted positive unique indices, fill mode
    input_dict = {
        "a": np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        "indices": np.array([1, 2], dtype=np.int32),
        "axis": 0,
        "mode": "fill",
        "unique_indices": True,
        "indices_are_sorted": True,
        "fill_value": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array along axis 0, clip mode
    input_dict = {
        "a": np.arange(12, dtype=np.float64).reshape(3, 4),
        "indices": np.array([0, 2], dtype=np.int32),
        "axis": 0,
        "mode": "clip",
        "unique_indices": True,
        "indices_are_sorted": True,
        "fill_value": -1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array along axis 1, negative indices
    input_dict = {
        "a": np.random.randn(4, 5).astype(np.float32),
        "indices": np.array([-1, 0, 2], dtype=np.int32),
        "axis": 1,
        "mode": "fill",
        "unique_indices": False,
        "indices_are_sorted": False,
        "fill_value": float("nan"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Negative axis, mode clip
    input_dict = {
        "a": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        "indices": np.array([2, 1, 0], dtype=np.int32),
        "axis": -1,
        "mode": "clip",
        "unique_indices": True,
        "indices_are_sorted": False,
        "fill_value": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array along axis 1, non-unique indices
    input_dict = {
        "a": np.ones((4, 6, 2), dtype=np.float64),
        "indices": np.array([1, 1, 3, 3], dtype=np.int32),
        "axis": 1,
        "mode": "fill",
        "unique_indices": False,
        "indices_are_sorted": True,
        "fill_value": -999.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Out of bounds indices with mode fill and custom fill_value
    input_dict = {
        "a": np.array([1.5, 2.5, 3.5], dtype=np.float32),
        "indices": np.array([0, 5, 2], dtype=np.int32),
        "axis": 0,
        "mode": "fill",
        "unique_indices": True,
        "indices_are_sorted": False,
        "fill_value": 42.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Single index
    input_dict = {
        "a": np.arange(10, dtype=np.float32),
        "indices": np.array([3], dtype=np.int32),
        "axis": 0,
        "mode": "clip",
        "unique_indices": True,
        "indices_are_sorted": True,
        "fill_value": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 4D array along axis 2
    input_dict = {
        "a": np.random.uniform(-5.0, 5.0, size=(2, 3, 4, 5)).astype(np.float64),
        "indices": np.array([0, 3], dtype=np.int32),
        "axis": 2,
        "mode": "fill",
        "unique_indices": True,
        "indices_are_sorted": True,
        "fill_value": 1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Negative axis on 2D array, clip mode with repeated elements
    input_dict = {
        "a": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "indices": np.array([1, 0, 1], dtype=np.int32),
        "axis": -2,
        "mode": "clip",
        "unique_indices": False,
        "indices_are_sorted": False,
        "fill_value": 0.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Multi-element sorted indices on 3D array
    input_dict = {
        "a": np.arange(60, dtype=np.float32).reshape(3, 4, 5),
        "indices": np.array([0, 1, 2], dtype=np.int32),
        "axis": 1,
        "mode": "fill",
        "unique_indices": True,
        "indices_are_sorted": True,
        "fill_value": -1.0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.take_4"] = take_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.take_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.take_4'.")


check_valid('jax.numpy.take', generated_inputs['jax.numpy.take_4'], lib="jax", suffix=4)
