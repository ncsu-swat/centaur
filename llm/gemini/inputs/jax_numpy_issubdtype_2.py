
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def issubdtype_inputs():
    list_of_inputs = []

    # Input 1: float32 with generic floating
    input_dict = {
        "arg1": "float32",
        "arg2": np.floating
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2: int32 with generic integer
    input_dict = {
        "arg1": "int32",
        "arg2": np.integer
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 3: uint8 with generic unsignedinteger
    input_dict = {
        "arg1": "uint8",
        "arg2": np.unsignedinteger
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 4: complex64 with generic complexfloating
    input_dict = {
        "arg1": "complex64",
        "arg2": np.complexfloating
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 5: bool with generic bool_
    input_dict = {
        "arg1": "bool",
        "arg2": np.bool_
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 6: float64 with exact float64 dtype
    input_dict = {
        "arg1": "float64",
        "arg2": np.dtype('float64')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 7: int16 with exact int16 dtype
    input_dict = {
        "arg1": "int16",
        "arg2": np.dtype('int16')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 8: uint32 with exact uint32 dtype
    input_dict = {
        "arg1": "uint32",
        "arg2": np.dtype('uint32')
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 9: complex128 with generic number
    input_dict = {
        "arg1": "complex128",
        "arg2": np.number
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 10: int8 with generic signedinteger
    input_dict = {
        "arg1": "int8",
        "arg2": np.signedinteger
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["jax.numpy.issubdtype_2"] = issubdtype_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.issubdtype_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.issubdtype_2'.")


check_valid('jax.numpy.issubdtype', generated_inputs['jax.numpy.issubdtype_2'], lib="jax", suffix=2)
