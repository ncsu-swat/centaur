
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def array_inputs():
    list_of_inputs = []

    # Input 1: 1D integer list, basic case
    input_dict = {
        "object": [1, 2, 3, 4, 5],
        "dtype": np.dtype("int32"),
        "copy": True,
        "order": "K",
        "ndmin": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: 1D float list with negative values, copy=False, ndmin=1
    input_dict = {
        "object": [-1.5, -2.0, 3.5, 4.0],
        "dtype": np.dtype("float32"),
        "copy": False,
        "order": "K",
        "ndmin": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: 2D float list, ndmin=2
    input_dict = {
        "object": [[1.0, 2.0], [3.0, 4.0]],
        "dtype": np.dtype("float64"),
        "copy": True,
        "order": "K",
        "ndmin": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Boolean list, converting to int dtype
    input_dict = {
        "object": [True, False, True],
        "dtype": np.dtype("int8"),
        "copy": True,
        "order": "K",
        "ndmin": 0
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Float list converting to boolean dtype, ndmin=3
    input_dict = {
        "object": [[[0.0, 1.0], [1.0, 0.0]]],
        "dtype": np.dtype("bool"),
        "copy": False,
        "order": "K",
        "ndmin": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Complex values in nested list
    input_dict = {
        "object": [[1 + 2j, 3 - 4j]],
        "dtype": np.dtype("complex64"),
        "copy": True,
        "order": "K",
        "ndmin": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Empty list, high ndmin
    input_dict = {
        "object": [],
        "dtype": np.dtype("float32"),
        "copy": True,
        "order": "K",
        "ndmin": 4
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Deeply nested float16 list
    input_dict = {
        "object": [[[[1.0], [2.0]], [[3.0], [4.0]]]],
        "dtype": np.dtype("float16"),
        "copy": False,
        "order": "K",
        "ndmin": 2
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large positive and negative integers, int64 dtype
    input_dict = {
        "object": [100000, -200000, 300000],
        "dtype": np.dtype("int64"),
        "copy": True,
        "order": "K",
        "ndmin": 1
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Unsigned integer list, uint32 dtype
    input_dict = {
        "object": [10, 20, 30, 40],
        "dtype": np.dtype("uint32"),
        "copy": True,
        "order": "K",
        "ndmin": 3
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.array_2"] = array_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.array_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.array_2'.")


check_valid('jax.numpy.array', generated_inputs['jax.numpy.array_2'], lib="jax", suffix=2)
