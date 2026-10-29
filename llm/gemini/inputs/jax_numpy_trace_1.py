
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def trace_inputs():
    list_of_inputs = []

    # Input 1: Basic 2D float32 array
    input_dict = {
        "a": np.arange(9, dtype=np.float32).reshape(3, 3),
        "offset": 0,
        "axis1": 0,
        "axis2": 1,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D rectangular matrix with positive offset
    input_dict = {
        "a": np.random.randn(4, 6).astype(np.float64),
        "offset": 1,
        "axis1": 0,
        "axis2": 1,
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D matrix with negative offset
    input_dict = {
        "a": np.random.randint(-10, 10, size=(5, 5)).astype(np.int32),
        "offset": -2,
        "axis1": 0,
        "axis2": 1,
        "dtype": np.dtype(np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D tensor with axis permutation
    input_dict = {
        "a": np.random.randn(2, 3, 4).astype(np.float32),
        "offset": 0,
        "axis1": 1,
        "axis2": 2,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D tensor with negative axis indexing
    input_dict = {
        "a": np.random.randn(4, 5, 6).astype(np.float64),
        "offset": -1,
        "axis1": -2,
        "axis2": -1,
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D tensor with different axes
    input_dict = {
        "a": np.random.randn(2, 3, 4, 5).astype(np.float32),
        "offset": 0,
        "axis1": 0,
        "axis2": 2,
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Complex data type
    input_dict = {
        "a": (np.random.randn(3, 3) + 1j * np.random.randn(3, 3)).astype(
            np.complex64
        ),
        "offset": 0,
        "axis1": 0,
        "axis2": 1,
        "dtype": np.dtype(np.complex64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Integer type with positive offset
    input_dict = {
        "a": np.ones((6, 6), dtype=np.int64),
        "offset": 3,
        "axis1": 1,
        "axis2": 0,
        "dtype": np.dtype(np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 5D tensor
    input_dict = {
        "a": np.random.randn(2, 2, 3, 3, 2).astype(np.float32),
        "offset": 1,
        "axis1": 2,
        "axis2": 3,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Boolean array casting to int32 dtype
    input_dict = {
        "a": np.eye(4, dtype=np.bool_),
        "offset": 0,
        "axis1": 0,
        "axis2": 1,
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D array with axis1 > axis2
    input_dict = {
        "a": np.random.randn(3, 4, 3).astype(np.float32),
        "offset": 0,
        "axis1": 2,
        "axis2": 0,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.trace_1"] = trace_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.trace_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.trace_1'.")


check_valid('jax.numpy.trace', generated_inputs['jax.numpy.trace_1'], lib="jax", suffix=1)
