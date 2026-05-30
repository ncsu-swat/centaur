
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def cumsum_inputs():
    list_of_inputs = []

    # Input 1: 1D integer array, axis 0
    a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    input_dict = {
        "a": a,
        "axis": 0,
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float array, axis 0
    a = np.random.randn(3, 4).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": 0,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float array, axis 1, casting to float64
    a = np.random.randn(5, 2).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": 1,
        "dtype": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D integer array with negative values, axis 2
    a = np.random.randint(-10, 10, size=(2, 3, 4)).astype(np.int32)
    input_dict = {
        "a": a,
        "axis": 2,
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D float array, negative axis
    a = np.random.randn(2, 2, 2).astype(np.float64)
    input_dict = {
        "a": a,
        "axis": -1,
        "dtype": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float array, axis 3, casting to float32
    a = np.random.randn(2, 3, 2, 4).astype(np.float16)
    input_dict = {
        "a": a,
        "axis": 3,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 1D array with large integers, casting to int64 to avoid overflow
    a = np.array([1000000, 2000000, 3000000], dtype=np.int32)
    input_dict = {
        "a": a,
        "axis": 0,
        "dtype": np.int64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 2D array, negative axis -2
    a = np.random.randn(4, 3).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": -2,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 3D array with zeros, axis 1
    a = np.zeros((3, 3, 3), dtype=np.int32)
    input_dict = {
        "a": a,
        "axis": 1,
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D float array, axis 1
    a = np.random.randn(2, 4, 3, 2).astype(np.float32)
    input_dict = {
        "a": a,
        "axis": 1,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.cumsum"] = cumsum_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.cumsum' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.cumsum'.")


check_valid('jax.numpy.cumsum', generated_inputs['jax.numpy.cumsum'], lib="jax", suffix=0)
