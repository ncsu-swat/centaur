
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def trace_inputs():
    list_of_inputs = []

    # Input 1: Simple 2D float32 square matrix, main diagonal
    input_dict = {
        "a": np.random.randn(4, 4).astype(np.float32),
        "offset": 0,
        "axis1": 0,
        "axis2": 1,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D float64 square matrix, positive offset
    input_dict = {
        "a": np.random.randn(5, 5).astype(np.float64),
        "offset": 1,
        "axis1": 0,
        "axis2": 1,
        "dtype": np.float64
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D int32 matrix, negative offset
    input_dict = {
        "a": np.random.randint(-10, 10, size=(4, 6)).astype(np.int32),
        "offset": -2,
        "axis1": 0,
        "axis2": 1,
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 3D float32 array, trace along axis1=1, axis2=2
    input_dict = {
        "a": np.random.randn(2, 3, 3).astype(np.float32),
        "offset": 0,
        "axis1": 1,
        "axis2": 2,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D int32 array, trace along axis1=0, axis2=2 with offset
    input_dict = {
        "a": np.random.randint(-5, 5, size=(3, 4, 3)).astype(np.int32),
        "offset": 1,
        "axis1": 0,
        "axis2": 2,
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 4D float32 array, trace along last two dimensions
    input_dict = {
        "a": np.random.randn(2, 2, 4, 4).astype(np.float32),
        "offset": -1,
        "axis1": 2,
        "axis2": 3,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 2D float32 matrix with offset
    input_dict = {
        "a": np.random.randn(6, 6).astype(np.float32),
        "offset": 2,
        "axis1": 0,
        "axis2": 1,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D float32 array, trace along axis1=0, axis2=1
    input_dict = {
        "a": np.random.randn(4, 4, 2).astype(np.float32),
        "offset": -1,
        "axis1": 0,
        "axis2": 1,
        "dtype": np.float32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D int32 matrix
    input_dict = {
        "a": np.random.randint(-10, 10, size=(5, 5)).astype(np.int32),
        "offset": 0,
        "axis1": 0,
        "axis2": 1,
        "dtype": np.int32
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 4D float64 array, trace along axis1=0, axis2=3
    input_dict = {
        "a": np.random.randn(3, 2, 2, 3).astype(np.float64),
        "offset": 0,
        "axis1": 0,
        "axis2": 3,
        "dtype": np.float64
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
