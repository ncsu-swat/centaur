
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def cumprod_inputs():
    list_of_inputs = []

    # Input 1: 1D float array, axis 0
    input_dict = {
        "a": np.array([1.5, -2.0, 3.0, 0.5], dtype=np.float32),
        "axis": 0,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D integer array, axis 0
    input_dict = {
        "a": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
        "axis": 0,
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D integer array, axis 1, cast to int64
    input_dict = {
        "a": np.array([[2, -3, 4], [-1, 5, -2]], dtype=np.int32),
        "axis": 1,
        "dtype": np.dtype(np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D float array with negative axis
    input_dict = {
        "a": np.array(
            [[0.5, 1.2, -3.4], [2.1, -0.4, 1.0]], dtype=np.float64
        ),
        "axis": -1,
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float array, axis 1
    input_dict = {
        "a": np.random.randn(2, 3, 4).astype(np.float32),
        "axis": 1,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D complex array, axis 2
    input_dict = {
        "a": (
            np.random.randn(2, 2, 3) + 1j * np.random.randn(2, 2, 3)
        ).astype(np.complex64),
        "axis": 2,
        "dtype": np.dtype(np.complex64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D float array, axis -2
    input_dict = {
        "a": np.random.uniform(0.5, 1.5, size=(2, 3, 2, 4)).astype(np.float32),
        "axis": -2,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 1D boolean array accumulated to int32
    input_dict = {
        "a": np.array([True, True, False, True], dtype=np.bool_),
        "axis": 0,
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D float array with zeros, axis 0
    input_dict = {
        "a": np.array([[1.0, 0.0, 3.0], [4.0, 5.0, 0.0]], dtype=np.float32),
        "axis": 0,
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 1D array of single element
    input_dict = {
        "a": np.array([42], dtype=np.int32),
        "axis": 0,
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D integer array with negative values, axis -3
    input_dict = {
        "a": np.array(
            [[[1, -2], [3, -4]], [[-5, 6], [-7, 8]]], dtype=np.int16
        ),
        "axis": -3,
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.cumprod"] = cumprod_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.cumprod' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.cumprod'.")


check_valid('jax.numpy.cumprod', generated_inputs['jax.numpy.cumprod'], lib="jax", suffix=0)
