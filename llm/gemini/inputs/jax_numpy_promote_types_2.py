
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import numpy as np
import copy

def promote_types_inputs():
    list_of_inputs = []

    # Input 1: int32 and float32
    list_of_inputs.append({"a": "int32", "b": "float32"})

    # Input 2: float16 and float32
    list_of_inputs.append({"a": "float16", "b": "float32"})

    # Input 3: int64 and float64
    list_of_inputs.append({"a": "int64", "b": "float64"})

    # Input 4: uint8 and int16
    list_of_inputs.append({"a": "uint8", "b": "int16"})

    # Input 5: bool and int8
    list_of_inputs.append({"a": "bool", "b": "int8"})

    # Input 6: complex64 and float32
    list_of_inputs.append({"a": "complex64", "b": "float32"})

    # Input 7: float64 and complex128
    list_of_inputs.append({"a": "float64", "b": "complex128"})

    # Input 8: uint32 and int64
    list_of_inputs.append({"a": "uint32", "b": "int64"})

    # Input 9: bool and bool
    list_of_inputs.append({"a": "bool", "b": "bool"})

    # Input 10: int32 and int32
    list_of_inputs.append({"a": "int32", "b": "int32"})

    # Input 11: float16 and complex64
    list_of_inputs.append({"a": "float16", "b": "complex64"})

    # Input 12: int16 and float64
    list_of_inputs.append({"a": "int16", "b": "float64"})

    return list_of_inputs

generated_inputs["jax.numpy.promote_types_2"] = promote_types_inputs()

def check_valid(api, list_of_inputs, lib="jax", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'jax.numpy.promote_types_2' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'jax.numpy.promote_types_2'.")


check_valid('jax.numpy.promote_types', generated_inputs['jax.numpy.promote_types_2'], lib="jax", suffix=2)
