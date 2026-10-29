
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def argsort_inputs():
    list_of_inputs = []

    # Input 1: 1D float32 array, axis 0
    input_dict = {
        "a": np.array([3.14, -1.5, 0.0, 2.71], dtype=np.float32),
        "axis": 0,
        "stable": True,
        "descending": False,
        "dtype": np.dtype("int32"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float64 array, axis -1, descending
    input_dict = {
        "a": np.array([-10.0, 50.5, 20.2, -30.3], dtype=np.float64),
        "axis": -1,
        "stable": False,
        "descending": True,
        "dtype": np.dtype("int64"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float32 array, axis 1
    input_dict = {
        "a": np.array([[1.0, 4.0, 2.0], [5.0, 0.0, 3.0]], dtype=np.float32),
        "axis": 1,
        "stable": True,
        "descending": False,
        "dtype": np.dtype("int32"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float64 array, axis 0, descending
    input_dict = {
        "a": np.array([[-1.2, 3.4], [5.6, -7.8], [0.0, 2.1]], dtype=np.float64),
        "axis": 0,
        "stable": False,
        "descending": True,
        "dtype": np.dtype("int32"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 2D float32 array, negative axis -2
    input_dict = {
        "a": np.array([[10.0, 20.0], [5.0, 25.0], [15.0, 1.0]], dtype=np.float32),
        "axis": -2,
        "stable": True,
        "descending": True,
        "dtype": np.dtype("int64"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D float32 array, axis 2
    input_dict = {
        "a": np.arange(24, dtype=np.float32).reshape(2, 3, 4)[::-1],
        "axis": 2,
        "stable": True,
        "descending": False,
        "dtype": np.dtype("int32"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 3D float64 array, axis -1, descending
    input_dict = {
        "a": np.array(
            [[[1.5, -2.5], [3.0, 0.5]], [[-1.0, 4.2], [2.1, -3.3]]],
            dtype=np.float64,
        ),
        "axis": -1,
        "stable": False,
        "descending": True,
        "dtype": np.dtype("int64"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float32 array with inf and nan values, axis 1
    input_dict = {
        "a": np.array(
            [[[np.nan, 1.0], [np.inf, -np.inf]], [[0.0, 2.0], [-1.0, 5.0]]],
            dtype=np.float32,
        ),
        "axis": 1,
        "stable": True,
        "descending": False,
        "dtype": np.dtype("int32"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 1D array of single element, axis 0
    input_dict = {
        "a": np.array([42.0], dtype=np.float32),
        "axis": 0,
        "stable": False,
        "descending": False,
        "dtype": np.dtype("int32"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D float32 array, axis -3
    input_dict = {
        "a": np.ones((2, 3, 2, 2), dtype=np.float32)
        * np.array([-1.0, 2.0]).reshape(1, 1, 1, 2),
        "axis": -3,
        "stable": True,
        "descending": True,
        "dtype": np.dtype("int64"),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.argsort_3"] = argsort_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.argsort_3' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.argsort_3'.")


check_valid('jax.numpy.argsort', generated_inputs['jax.numpy.argsort_3'], lib="jax", suffix=3)
