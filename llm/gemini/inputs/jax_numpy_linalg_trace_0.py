
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def trace_inputs():
    list_of_inputs = []

    # Input 1: 2D square float32 matrix, default offset
    x = np.random.randn(4, 4).astype(np.float32)
    input_dict = {
        "x": x,
        "offset": 0,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D rectangular int32 matrix with positive offset
    x = np.random.randint(-10, 10, size=(3, 5)).astype(np.int32)
    input_dict = {
        "x": x,
        "offset": 1,
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D rectangular float64 matrix with negative offset
    x = np.random.randn(5, 3).astype(np.float64)
    input_dict = {
        "x": x,
        "offset": -1,
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D batched square matrices, offset 0, type promotion to float64
    x = np.random.randn(2, 3, 3).astype(np.float32)
    input_dict = {
        "x": x,
        "offset": 0,
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D batched rectangular int64 matrices, negative offset
    x = np.random.randint(-100, 100, size=(4, 6, 4)).astype(np.int64)
    input_dict = {
        "x": x,
        "offset": -2,
        "dtype": np.dtype(np.int64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D batched complex64 matrices with positive offset
    x = (np.random.randn(2, 2, 4, 4) + 1j * np.random.randn(2, 2, 4, 4)).astype(
        np.complex64
    )
    input_dict = {
        "x": x,
        "offset": 2,
        "dtype": np.dtype(np.complex64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D batched complex128 matrices with negative offset
    x = (np.random.randn(3, 1, 5, 5) + 1j * np.random.randn(3, 1, 5, 5)).astype(
        np.complex128
    )
    input_dict = {
        "x": x,
        "offset": -1,
        "dtype": np.dtype(np.complex128),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 5D batched float32 matrices with offset 0
    x = np.random.randn(2, 1, 3, 3, 3).astype(np.float32)
    input_dict = {
        "x": x,
        "offset": 0,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D boolean matrix cast to int32 dtype
    x = np.random.choice([True, False], size=(6, 6))
    input_dict = {
        "x": x,
        "offset": 0,
        "dtype": np.dtype(np.int32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D matrix with large positive offset
    x = np.random.randn(8, 8).astype(np.float64)
    input_dict = {
        "x": x,
        "offset": 5,
        "dtype": np.dtype(np.float64),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: 3D batched rectangular matrices with large negative offset
    x = np.random.randn(3, 7, 5).astype(np.float32)
    input_dict = {
        "x": x,
        "offset": -4,
        "dtype": np.dtype(np.float32),
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.linalg.trace"] = trace_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.linalg.trace' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.linalg.trace'.")


check_valid('jax.numpy.linalg.trace', generated_inputs['jax.numpy.linalg.trace'], lib="jax", suffix=0)
