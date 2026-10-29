
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def take_inputs():
    list_of_inputs = []

    # Input 1: 1D boolean array, unique and sorted indices
    a = np.array([True, False, True, False, True], dtype=bool)
    indices = np.array([0, 2, 4], dtype=np.int32)
    input_dict = {
        "a": a,
        "indices": indices,
        "axis": 0,
        "mode": "fill",
        "unique_indices": True,
        "indices_are_sorted": True,
        "fill_value": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float array, non-sorted duplicate indices with mode='clip'
    a = np.random.randn(4, 5).astype(np.float32)
    indices = np.array([3, 1, 1, 2], dtype=np.int32)
    input_dict = {
        "a": a,
        "indices": indices,
        "axis": 1,
        "mode": "clip",
        "unique_indices": False,
        "indices_are_sorted": False,
        "fill_value": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D int array, axis=0, sorted unique indices
    a = np.arange(12, dtype=np.int32).reshape(3, 4)
    indices = np.array([0, 1], dtype=np.int32)
    input_dict = {
        "a": a,
        "indices": indices,
        "axis": 0,
        "mode": "fill",
        "unique_indices": True,
        "indices_are_sorted": True,
        "fill_value": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float array, negative axis, mode='clip'
    a = np.random.randn(2, 3, 4).astype(np.float64)
    indices = np.array([1, 2, 3], dtype=np.int32)
    input_dict = {
        "a": a,
        "indices": indices,
        "axis": -1,
        "mode": "clip",
        "unique_indices": True,
        "indices_are_sorted": True,
        "fill_value": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int array, sorted indices with duplicates
    a = np.random.randint(0, 10, size=(3, 6, 2), dtype=np.int64)
    indices = np.array([0, 2, 2, 5], dtype=np.int32)
    input_dict = {
        "a": a,
        "indices": indices,
        "axis": 1,
        "mode": "fill",
        "unique_indices": False,
        "indices_are_sorted": True,
        "fill_value": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D float array, negative axis=-1, non-unique non-sorted
    a = np.array([1.5, -2.3, 4.2, 0.0], dtype=np.float32)
    indices = np.array([2, 0, 3, 0], dtype=np.int32)
    input_dict = {
        "a": a,
        "indices": indices,
        "axis": -1,
        "mode": "clip",
        "unique_indices": False,
        "indices_are_sorted": False,
        "fill_value": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D boolean array, unique non-sorted indices
    a = np.ones((2, 3, 4, 2), dtype=bool)
    indices = np.array([3, 1, 0], dtype=np.int32)
    input_dict = {
        "a": a,
        "indices": indices,
        "axis": 2,
        "mode": "fill",
        "unique_indices": True,
        "indices_are_sorted": False,
        "fill_value": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D float array, negative axis=-2, sorted unique indices
    a = np.random.randn(5, 3).astype(np.float32)
    indices = np.array([1, 3, 4], dtype=np.int32)
    input_dict = {
        "a": a,
        "indices": indices,
        "axis": -2,
        "mode": "clip",
        "unique_indices": True,
        "indices_are_sorted": True,
        "fill_value": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D int array, axis=0, unique non-sorted indices
    a = np.arange(24, dtype=np.int32).reshape(4, 3, 2)
    indices = np.array([2, 0, 3], dtype=np.int32)
    input_dict = {
        "a": a,
        "indices": indices,
        "axis": 0,
        "mode": "fill",
        "unique_indices": True,
        "indices_are_sorted": False,
        "fill_value": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D int array with single index
    a = np.array([-10, 20, -30, 40], dtype=np.int16)
    indices = np.array([2], dtype=np.int32)
    input_dict = {
        "a": a,
        "indices": indices,
        "axis": 0,
        "mode": "clip",
        "unique_indices": True,
        "indices_are_sorted": True,
        "fill_value": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 2D float array with multi-dimensional indices
    a = np.random.randn(3, 4).astype(np.float32)
    indices = np.array([[0, 1], [2, 3]], dtype=np.int32)
    input_dict = {
        "a": a,
        "indices": indices,
        "axis": 1,
        "mode": "fill",
        "unique_indices": False,
        "indices_are_sorted": False,
        "fill_value": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.take_3"] = take_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.take_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.take_3'.")


check_valid('jax.numpy.take', generated_inputs['jax.numpy.take_3'], lib="jax", suffix=3)
