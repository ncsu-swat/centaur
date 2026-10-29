
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def array_inputs():
    list_of_inputs = []

    # Input 1: Zero scalar with int32 dtype and ndmin 0
    input_dict = {
        "object": 0,
        "dtype": np.dtype("int32"),
        "copy": True,
        "order": "K",
        "ndmin": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: Positive integer to float32 with ndmin 1
    input_dict = {
        "object": 42,
        "dtype": np.dtype("float32"),
        "copy": False,
        "order": "K",
        "ndmin": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: Negative integer to int64 with ndmin 2
    input_dict = {
        "object": -15,
        "dtype": np.dtype("int64"),
        "copy": True,
        "order": "K",
        "ndmin": 2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: Integer converted to boolean dtype
    input_dict = {
        "object": 1,
        "dtype": np.dtype("bool"),
        "copy": False,
        "order": "K",
        "ndmin": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: Value within uint8 range with ndmin 3
    input_dict = {
        "object": 255,
        "dtype": np.dtype("uint8"),
        "copy": True,
        "order": "K",
        "ndmin": 3,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: Negative integer to float64 with ndmin 1
    input_dict = {
        "object": -1024,
        "dtype": np.dtype("float64"),
        "copy": False,
        "order": "K",
        "ndmin": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: Positive integer to int16 with ndmin 4
    input_dict = {
        "object": 7,
        "dtype": np.dtype("int16"),
        "copy": True,
        "order": "K",
        "ndmin": 4,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: Boundary value for int8 with ndmin 2
    input_dict = {
        "object": -128,
        "dtype": np.dtype("int8"),
        "copy": False,
        "order": "K",
        "ndmin": 2,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: Large integer to uint32 with ndmin 0
    input_dict = {
        "object": 65535,
        "dtype": np.dtype("uint32"),
        "copy": True,
        "order": "K",
        "ndmin": 0,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: Large negative integer to int32 with higher dimensionality ndmin 5
    input_dict = {
        "object": -999999,
        "dtype": np.dtype("int32"),
        "copy": False,
        "order": "K",
        "ndmin": 5,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: Very large positive integer to uint64 with ndmin 1
    input_dict = {
        "object": 123456789,
        "dtype": np.dtype("uint64"),
        "copy": True,
        "order": "K",
        "ndmin": 1,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.array_4"] = array_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.array_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.array_4'.")


check_valid('jax.numpy.array', generated_inputs['jax.numpy.array_4'], lib="jax", suffix=4)
