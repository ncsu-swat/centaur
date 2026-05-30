
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def jax_numpy_sort_inputs():
    list_of_inputs = []

    # Input 1: 1D array of integers
    input_dict = {
        "a": np.array([5, 1, 9, 3, 7], dtype=np.int32),
        "axis": -1,
        "stable": True,
        "descending": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D array of floats with negative values
    input_dict = {
        "a": np.array([-1.5, 3.2, 0.0, -0.5, 2.1], dtype=np.float32),
        "axis": 0,
        "stable": False,
        "descending": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D array of integers, axis=0
    input_dict = {
        "a": np.array([[3, 2, 1], [6, 5, 4]], dtype=np.int32),
        "axis": 0,
        "stable": True,
        "descending": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array of integers, axis=1
    input_dict = {
        "a": np.array([[3, 2, 1], [6, 5, 4]], dtype=np.int32),
        "axis": 1,
        "stable": True,
        "descending": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array of floats
    input_dict = {
        "a": np.array([[[1.2, 0.5], [3.1, 2.2]], [[-1.1, -0.5], [0.1, 0.9]]], dtype=np.float32),
        "axis": -1,
        "stable": True,
        "descending": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 1D array of negative integers
    input_dict = {
        "a": np.array([-10, -20, -5, -1, -15], dtype=np.int32),
        "axis": -1,
        "stable": False,
        "descending": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D array with single element columns
    input_dict = {
        "a": np.array([[10], [5], [20]], dtype=np.int32),
        "axis": 0,
        "stable": True,
        "descending": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D array, axis=2
    input_dict = {
        "a": np.array([[[5, 3], [1, 2]], [[9, 7], [4, 6]]], dtype=np.int32),
        "axis": 2,
        "stable": True,
        "descending": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D array of float values, unstable sort, descending
    input_dict = {
        "a": np.array([[0.5, -0.5, 1.5], [2.5, 1.2, -1.2]], dtype=np.float32),
        "axis": -1,
        "stable": False,
        "descending": True
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array of float numbers, sorting ascending
    input_dict = {
        "a": np.array([1.0, 2.5, -3.0, 4.2, 0.0], dtype=np.float32),
        "axis": 0,
        "stable": True,
        "descending": False
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.sort_2"] = jax_numpy_sort_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sort_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sort_2'.")


check_valid('jax.numpy.sort', generated_inputs['jax.numpy.sort_2'], lib="jax", suffix=2)
