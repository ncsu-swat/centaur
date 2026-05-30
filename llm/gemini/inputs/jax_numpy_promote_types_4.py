
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def promote_types_inputs():
    list_of_inputs = []

    # Input 1: int32 string and float32 dtype
    input_dict = {
        "a": "int32",
        "b": np.dtype("float32")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: float16 string and float64 dtype
    input_dict = {
        "a": "float16",
        "b": np.dtype("float64")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: uint8 string and int16 dtype
    input_dict = {
        "a": "uint8",
        "b": np.dtype("int16")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: bool string and int32 dtype
    input_dict = {
        "a": "bool",
        "b": np.dtype("int32")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: complex64 string and float32 dtype
    input_dict = {
        "a": "complex64",
        "b": np.dtype("float32")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float32 string and complex128 dtype
    input_dict = {
        "a": "float32",
        "b": np.dtype("complex128")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int64 string and uint32 dtype
    input_dict = {
        "a": "int64",
        "b": np.dtype("uint32")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: uint16 string and float16 dtype
    input_dict = {
        "a": "uint16",
        "b": np.dtype("float16")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: int8 string and bool dtype
    input_dict = {
        "a": "int8",
        "b": np.dtype("bool")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: float64 string and int8 dtype
    input_dict = {
        "a": "float64",
        "b": np.dtype("int8")
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.promote_types_4"] = promote_types_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.promote_types_4' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.promote_types_4'.")


check_valid('jax.numpy.promote_types', generated_inputs['jax.numpy.promote_types_4'], lib="jax", suffix=4)
