
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np


def asarray_inputs():
    list_of_inputs = []

    # Input 1: zero integer with int32 dtype
    input_dict = {"a": 0, "dtype": np.dtype(np.int32), "order": "K", "copy": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: positive integer with float32 dtype
    input_dict = {"a": 1, "dtype": np.dtype(np.float32), "order": "K", "copy": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: negative integer with int64 dtype
    input_dict = {"a": -5, "dtype": np.dtype(np.int64), "order": "K", "copy": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: positive integer with float64 dtype
    input_dict = {"a": 42, "dtype": np.dtype(np.float64), "order": "K", "copy": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: boundary unsigned 8-bit integer with uint8 dtype
    input_dict = {"a": 255, "dtype": np.dtype(np.uint8), "order": "K", "copy": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: negative 8-bit boundary integer with int8 dtype
    input_dict = {
        "a": -128,
        "dtype": np.dtype(np.int8),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: positive 16-bit integer with int16 dtype
    input_dict = {
        "a": 1000,
        "dtype": np.dtype(np.int16),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: integer with bool dtype
    input_dict = {"a": 1, "dtype": np.dtype(np.bool_), "order": "K", "copy": True}
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: integer with complex64 dtype
    input_dict = {
        "a": 3,
        "dtype": np.dtype(np.complex64),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: negative integer with float16 dtype
    input_dict = {
        "a": -99,
        "dtype": np.dtype(np.float16),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 11: unsigned 16-bit integer with uint16 dtype
    input_dict = {
        "a": 65535,
        "dtype": np.dtype(np.uint16),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 12: unsigned 32-bit integer with uint32 dtype
    input_dict = {
        "a": 1000000,
        "dtype": np.dtype(np.uint32),
        "order": "K",
        "copy": True,
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs


generated_inputs["jax.numpy.asarray_4"] = asarray_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.asarray_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.asarray_4'.")


check_valid('jax.numpy.asarray', generated_inputs['jax.numpy.asarray_4'], lib="jax", suffix=4)
