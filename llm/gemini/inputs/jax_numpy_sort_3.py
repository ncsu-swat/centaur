
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def sort_inputs():
    list_of_inputs = []

    # Input 1: 1D int32 array, sort ascending
    input_dict = {
        "a": np.array([5, 2, 9, 1, 5, 6], dtype=np.int32),
        "axis": 0,
        "stable": True,
        "descending": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float32 array, sort descending
    input_dict = {
        "a": np.array([3.14, -1.5, 2.71, 0.0, -0.5], dtype=np.float32),
        "axis": -1,
        "stable": False,
        "descending": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 1D int64 array with negative numbers
    input_dict = {
        "a": np.array([-10, -3, -50, -1, -20], dtype=np.int64),
        "axis": 0,
        "stable": True,
        "descending": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D int32 array, sort along axis 0
    input_dict = {
        "a": np.array([[3, 2, 1], [6, 5, 4]], dtype=np.int32),
        "axis": 0,
        "stable": False,
        "descending": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D int32 array, sort along axis 1
    input_dict = {
        "a": np.array([[9, 4, 7], [1, 8, 3]], dtype=np.int32),
        "axis": 1,
        "stable": True,
        "descending": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 2D float64 array, sort along last axis descending
    input_dict = {
        "a": np.array([[-1.2, 3.4, 0.5], [2.1, -4.3, 1.1]], dtype=np.float64),
        "axis": -1,
        "stable": True,
        "descending": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float32 array, sort along axis -2
    input_dict = {
        "a": np.array([[10.0, 20.0], [5.0, 25.0], [1.0, 15.0]], dtype=np.float32),
        "axis": -2,
        "stable": False,
        "descending": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D int32 array, sort along axis 0
    input_dict = {
        "a": np.array([[[3, 2], [1, 4]], [[0, 5], [6, -1]]], dtype=np.int32),
        "axis": 0,
        "stable": True,
        "descending": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D float32 array, sort along axis 2 descending
    input_dict = {
        "a": np.array(
            [[[1.5, -2.5], [3.0, 0.0]], [[-1.0, 4.2], [2.3, -3.1]]],
            dtype=np.float32,
        ),
        "axis": 2,
        "stable": False,
        "descending": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D boolean array
    input_dict = {
        "a": np.array([True, False, True, False, True], dtype=bool),
        "axis": 0,
        "stable": True,
        "descending": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 1D single-element array
    input_dict = {
        "a": np.array([42], dtype=np.int32),
        "axis": -1,
        "stable": True,
        "descending": False,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.sort_3"] = sort_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.sort_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.sort_3'.")


check_valid('jax.numpy.sort', generated_inputs['jax.numpy.sort_3'], lib="jax", suffix=3)
