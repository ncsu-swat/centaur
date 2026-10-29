
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def asarray_inputs():
    list_of_inputs = []

    # Input 1: positive integer string with int32
    input_dict = {
        "a": "42",
        "dtype": np.dtype(np.int32),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: negative integer string with int32
    input_dict = {
        "a": "-1000",
        "dtype": np.dtype(np.int32),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: floating point string with float32
    input_dict = {
        "a": "3.14159",
        "dtype": np.dtype(np.float32),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: negative floating point string with float32
    input_dict = {
        "a": "-2.71828",
        "dtype": np.dtype(np.float32),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: scientific notation string with float32
    input_dict = {
        "a": "1e-5",
        "dtype": np.dtype(np.float32),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: positive exponent notation string with float32
    input_dict = {
        "a": "6.022e4",
        "dtype": np.dtype(np.float32),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: boolean string with bool dtype
    input_dict = {
        "a": "True",
        "dtype": np.dtype(np.bool_),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: 8-bit signed integer string
    input_dict = {
        "a": "-128",
        "dtype": np.dtype(np.int8),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: 8-bit unsigned integer string
    input_dict = {
        "a": "255",
        "dtype": np.dtype(np.uint8),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: 16-bit signed integer string
    input_dict = {
        "a": "32767",
        "dtype": np.dtype(np.int16),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: infinity string with float32
    input_dict = {
        "a": "inf",
        "dtype": np.dtype(np.float32),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: negative infinity string with float32
    input_dict = {
        "a": "-inf",
        "dtype": np.dtype(np.float32),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.asarray_7"] = asarray_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.asarray_7' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.asarray_7'.")


check_valid('jax.numpy.asarray', generated_inputs['jax.numpy.asarray_7'], lib="jax", suffix=7)
