
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def dstack_inputs():
    list_of_inputs = []

    # Input 1: 3D array (treated as list of 2D arrays)
    tup = np.random.randn(2, 3, 4).astype(np.float32)
    input_dict = {"tup": tup, "dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 2D array (treated as list of 1D arrays)
    tup = np.random.randn(3, 5).astype(np.float32)
    input_dict = {"tup": tup, "dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 3D array with negative/zero values, float64
    tup = np.array([[[1.0, -1.0], [0.0, 2.0]], [[-2.0, 0.5], [1.5, -0.5]]], dtype=np.float64)
    input_dict = {"tup": tup, "dtype": np.float64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: 2D array, int32
    tup = np.arange(12).reshape(3, 4).astype(np.int32)
    input_dict = {"tup": tup, "dtype": np.int32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: 3D array of ones, int32
    tup = np.ones((5, 2, 2), dtype=np.int32)
    input_dict = {"tup": tup, "dtype": np.int32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: 3D array of zeros, float32
    tup = np.zeros((2, 3, 3), dtype=np.float32)
    input_dict = {"tup": tup, "dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: 4D array, float32
    tup = np.random.randn(2, 2, 2, 2).astype(np.float32)
    input_dict = {"tup": tup, "dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 3D complex array, complex64
    tup = np.array([[[1+2j, 3+4j]], [[5+6j, 7+8j]]], dtype=np.complex64)
    input_dict = {"tup": tup, "dtype": np.complex64}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 2D boolean array, bool
    tup = np.array([[True, False], [False, True]])
    input_dict = {"tup": tup, "dtype": np.bool_}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 2D array, float32
    tup = np.random.randn(5, 4).astype(np.float32)
    input_dict = {"tup": tup, "dtype": np.float32}
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.dstack_1"] = dstack_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.dstack_1' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.dstack_1'.")


check_valid('jax.numpy.dstack', generated_inputs['jax.numpy.dstack_1'], lib="jax", suffix=1)
